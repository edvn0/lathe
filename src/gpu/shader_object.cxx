#include "gpu/shader_object.hxx"

#include <array>
#include <format>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/context.hxx"
#include "gpu/shader_binary_cache.hxx"
#include "gpu/vk_object_name.hxx"

namespace {
    auto make_error(ShaderObjectErrorType type, std::string_view message = {}, VkResult result = VK_SUCCESS,
                    std::source_location location = std::source_location::current()) noexcept -> ShaderObjectError {
        return ShaderObjectError{
                .type = type,
                .context =
                        ErrorContext{
                                .message = FlyString{message},
                                .vk_result = result != VK_SUCCESS ? std::optional{result} : std::nullopt,
                                .location = location,
                        },
        };
    }

    [[nodiscard]]
    auto create_layout(VulkanContext &context, std::span<VkDescriptorSetLayout const> additional_layouts,
                       std::span<VkPushConstantRange const> push_constant_ranges, VkDescriptorSetLayout global_layout,
                       std::string_view debug_name, VkPipelineLayout &out_layout) -> std::optional<ShaderObjectError> {
        auto layouts = std::vector<VkDescriptorSetLayout>{};

        layouts.push_back(global_layout);

        layouts.insert(layouts.end(), additional_layouts.begin(), additional_layouts.end());

        VkPipelineLayoutCreateInfo const layout_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .setLayoutCount = static_cast<std::uint32_t>(layouts.size()),
                .pSetLayouts = layouts.data(),
                .pushConstantRangeCount = static_cast<std::uint32_t>(push_constant_ranges.size()),
                .pPushConstantRanges = push_constant_ranges.data(),
        };

        auto const vk_result = vkCreatePipelineLayout(context.device, &layout_info, nullptr, &out_layout);

        if (vk_result != VK_SUCCESS) {
            return make_error(ShaderObjectErrorType::layout_creation_failed,
                              std::format("vkCreatePipelineLayout failed for shader object set '{}'", debug_name),
                              vk_result);
        }

        return std::nullopt;
    }
} // namespace

ShaderObjectSet::~ShaderObjectSet() { destroy(); }

ShaderObjectSet::ShaderObjectSet(ShaderObjectSet &&other) noexcept :
    context_(std::exchange(other.context_, nullptr)), shaders_(other.shaders_), stages_(other.stages_),
    count_(std::exchange(other.count_, 0)), layout_(std::exchange(other.layout_, VK_NULL_HANDLE)),
    bind_point_(std::exchange(other.bind_point_, VK_PIPELINE_BIND_POINT_GRAPHICS)) {
    other.shaders_.fill(VK_NULL_HANDLE);
}

auto ShaderObjectSet::operator=(ShaderObjectSet &&other) noexcept -> ShaderObjectSet & {
    if (this == &other) {
        return *this;
    }

    destroy();

    context_ = std::exchange(other.context_, nullptr);

    shaders_ = other.shaders_;
    stages_ = other.stages_;

    other.shaders_.fill(VK_NULL_HANDLE);

    count_ = std::exchange(other.count_, 0);

    layout_ = std::exchange(other.layout_, VK_NULL_HANDLE);

    bind_point_ = std::exchange(other.bind_point_, VK_PIPELINE_BIND_POINT_GRAPHICS);

    return *this;
}

auto ShaderObjectSet::create_linked(VulkanContext &context, ShaderObjectCreateInfo const &create_info,
                                    VkDescriptorSetLayout global_layout, ShaderBinaryCache const *binary_cache)
        -> std::expected<ShaderObjectSet, ShaderObjectError> {
    if (context.device == VK_NULL_HANDLE || create_info.shaders.empty()) {
        return std::unexpected(make_error(ShaderObjectErrorType::invalid_argument,
                                          context.device == VK_NULL_HANDLE ? "device is VK_NULL_HANDLE"
                                                                           : "no shader stages provided"));
    }

    if (create_info.shaders.size() > ShaderObjectSet::max_stages) {
        return std::unexpected(make_error(ShaderObjectErrorType::invalid_argument,
                                          std::format("too many shader stages ({}) for shader object set '{}'",
                                                      create_info.shaders.size(), create_info.debug_name)));
    }

    for (std::size_t index = 0; index < create_info.shaders.size(); ++index) {
        auto const &shader = create_info.shaders[index];

        if (shader.spirv.empty() || shader.entry_point.empty() ||
            shader.spirv.size_bytes() % sizeof(std::uint32_t) != 0) {
            return std::unexpected(make_error(ShaderObjectErrorType::invalid_argument,
                                              std::format("shader stage {} has invalid SPIR-V or entry point '{}'",
                                                          index, shader.entry_point.view())));
        }
    }

    ShaderObjectSet result;
    result.context_ = &context;

    if (auto layout_error =
                create_layout(context, create_info.additional_descriptor_set_layouts, create_info.push_constant_ranges,
                              global_layout, create_info.debug_name, result.layout_)) {
        result.context_ = nullptr;

        return std::unexpected(*layout_error);
    }

    auto layouts = std::vector<VkDescriptorSetLayout>{};
    layouts.push_back(global_layout);
    layouts.insert(layouts.end(), create_info.additional_descriptor_set_layouts.begin(),
                   create_info.additional_descriptor_set_layouts.end());

    std::vector<std::optional<std::vector<std::byte>>> binary_storage(create_info.shaders.size());
    std::vector<bool> used_binary(create_info.shaders.size(), false);

    auto const build_create_infos = [&](bool force_spirv) {
        std::vector<VkShaderCreateInfoEXT> infos(create_info.shaders.size());

        for (std::size_t index = 0; index < create_info.shaders.size(); ++index) {
            auto const &shader = create_info.shaders[index];
            auto const is_last = index + 1 == create_info.shaders.size();

            if (!force_spirv && binary_cache != nullptr && !shader.cache_key.empty()) {
                binary_storage[index] = binary_cache->find(shader.cache_key);
            } else {
                binary_storage[index].reset();
            }

            used_binary[index] = binary_storage[index].has_value();

            infos[index] = VkShaderCreateInfoEXT{
                    .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
                    .pNext = nullptr,
                    .flags = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT,
                    .stage = shader.stage,
                    .nextStage = is_last ? VkShaderStageFlags{0}
                                         : static_cast<VkShaderStageFlags>(create_info.shaders[index + 1].stage),
                    .codeType = used_binary[index] ? VK_SHADER_CODE_TYPE_BINARY_EXT : VK_SHADER_CODE_TYPE_SPIRV_EXT,
                    .codeSize = used_binary[index] ? binary_storage[index]->size() : shader.spirv.size_bytes(),
                    .pCode = used_binary[index] ? static_cast<void const *>(binary_storage[index]->data())
                                                : static_cast<void const *>(shader.spirv.data()),
                    .pName = shader.entry_point.view().data(),
                    .setLayoutCount = static_cast<std::uint32_t>(layouts.size()),
                    .pSetLayouts = layouts.data(),
                    .pushConstantRangeCount = static_cast<std::uint32_t>(create_info.push_constant_ranges.size()),
                    .pPushConstantRanges = create_info.push_constant_ranges.data(),
                    .pSpecializationInfo = shader.specialization_info,
            };
        }

        return infos;
    };

    auto shader_create_infos = build_create_infos(/*force_spirv=*/false);
    std::array<VkShaderEXT, ShaderObjectSet::max_stages> created_shaders{};
    auto vk_result = vkCreateShadersEXT(context.device, static_cast<std::uint32_t>(shader_create_infos.size()),
                                        shader_create_infos.data(), nullptr, created_shaders.data());

    if (vk_result == VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT) {
        for (auto const shader: created_shaders) {
            if (shader != VK_NULL_HANDLE) {
                vkDestroyShaderEXT(context.device, shader, nullptr);
            }
        }
        created_shaders.fill(VK_NULL_HANDLE);

        shader_create_infos = build_create_infos(/*force_spirv=*/true);

        vk_result = vkCreateShadersEXT(context.device, static_cast<std::uint32_t>(shader_create_infos.size()),
                                       shader_create_infos.data(), nullptr, created_shaders.data());
    }

    if (vk_result != VK_SUCCESS) {
        for (std::size_t index = 0; index < shader_create_infos.size(); ++index) {
            if (created_shaders[index] != VK_NULL_HANDLE) {
                vkDestroyShaderEXT(context.device, created_shaders[index], nullptr);
            }
        }

        vkDestroyPipelineLayout(context.device, result.layout_, nullptr);
        result.layout_ = VK_NULL_HANDLE;
        result.context_ = nullptr;

        return std::unexpected(
                make_error(ShaderObjectErrorType::shader_creation_failed,
                           std::format("vkCreateShadersEXT failed for shader object set '{}'", create_info.debug_name),
                           vk_result));
    }

    for (std::size_t index = 0; index < create_info.shaders.size(); ++index) {
        result.shaders_[index] = created_shaders[index];
        result.stages_[index] = create_info.shaders[index].stage;

        auto const object_name = std::format("{}.stage{}", create_info.debug_name, index);
        vk::set_object_name(context.device, VK_OBJECT_TYPE_SHADER_EXT, vk::object_handle(created_shaders[index]),
                            object_name);

        // Populate the cache for anything that didn't just come from it --
        // a first-time SPIR-V compile, or every stage after a retry (the
        // retry re-caches stages that weren't actually incompatible too;
        // harmless, just an extra vkGetShaderBinaryDataEXT call).
        auto const &shader = create_info.shaders[index];

        if (binary_cache != nullptr && !shader.cache_key.empty() && !used_binary[index]) {
            std::size_t size = 0;
            vkGetShaderBinaryDataEXT(context.device, created_shaders[index], &size, nullptr);

            if (size != 0) {
                std::vector<std::byte> blob(size);

                if (vkGetShaderBinaryDataEXT(context.device, created_shaders[index], &size, blob.data()) ==
                    VK_SUCCESS) {
                    blob.resize(size);
                    binary_cache->store(shader.cache_key, blob);
                }
            }
        }
    }

    result.count_ = static_cast<std::uint32_t>(create_info.shaders.size());
    result.bind_point_ = VK_PIPELINE_BIND_POINT_GRAPHICS;

    auto const layout_name = std::string{create_info.debug_name} + ".layout";
    vk::set_object_name(context.device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, vk::object_handle(result.layout_), layout_name);

    return result;
}

auto ShaderObjectSet::create_compute(VulkanContext &context, ComputeShaderCreateInfo const &create_info,
                                     VkDescriptorSetLayout global_layout, ShaderBinaryCache const *binary_cache)
        -> std::expected<ShaderObjectSet, ShaderObjectError> {
    if (context.device == VK_NULL_HANDLE || create_info.shader.spirv.empty()) {
        return std::unexpected(
                make_error(ShaderObjectErrorType::invalid_argument,
                           context.device == VK_NULL_HANDLE ? "device is VK_NULL_HANDLE" : "no shader stage provided"));
    }

    auto const &shader = create_info.shader;

    if (shader.entry_point.empty() || shader.spirv.size_bytes() % sizeof(std::uint32_t) != 0) {
        return std::unexpected(make_error(
                ShaderObjectErrorType::invalid_argument,
                std::format("compute shader has invalid SPIR-V or entry point '{}'", shader.entry_point.view())));
    }

    ShaderObjectSet result;
    result.context_ = &context;

    if (auto layout_error =
                create_layout(context, create_info.additional_descriptor_set_layouts, create_info.push_constant_ranges,
                              global_layout, create_info.debug_name, result.layout_)) {
        result.context_ = nullptr;

        return std::unexpected(*layout_error);
    }

    auto layouts = std::vector<VkDescriptorSetLayout>{};
    layouts.push_back(global_layout);
    layouts.insert(layouts.end(), create_info.additional_descriptor_set_layouts.begin(),
                   create_info.additional_descriptor_set_layouts.end());

    std::optional<std::vector<std::byte>> binary_storage;

    auto const build_create_info = [&](bool force_spirv) {
        if (!force_spirv && binary_cache != nullptr && !shader.cache_key.empty()) {
            binary_storage = binary_cache->find(shader.cache_key);
        } else {
            binary_storage.reset();
        }

        auto const use_binary = binary_storage.has_value();

        return VkShaderCreateInfoEXT{
                .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
                .pNext = nullptr,
                .flags = 0, // unlinked
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .nextStage = 0,
                .codeType = use_binary ? VK_SHADER_CODE_TYPE_BINARY_EXT : VK_SHADER_CODE_TYPE_SPIRV_EXT,
                .codeSize = use_binary ? binary_storage->size() : shader.spirv.size_bytes(),
                .pCode = use_binary ? static_cast<void const *>(binary_storage->data())
                                    : static_cast<void const *>(shader.spirv.data()),
                .pName = shader.entry_point.view().data(),
                .setLayoutCount = static_cast<std::uint32_t>(layouts.size()),
                .pSetLayouts = layouts.data(),
                .pushConstantRangeCount = static_cast<std::uint32_t>(create_info.push_constant_ranges.size()),
                .pPushConstantRanges = create_info.push_constant_ranges.data(),
                .pSpecializationInfo = shader.specialization_info,
        };
    };

    auto shader_create_info = build_create_info(/*force_spirv=*/false);
    auto used_binary = binary_storage.has_value();

    VkShaderEXT created_shader = VK_NULL_HANDLE;

    auto vk_result = vkCreateShadersEXT(context.device, 1, &shader_create_info, nullptr, &created_shader);

    if (vk_result == VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT) {
        if (created_shader != VK_NULL_HANDLE) {
            vkDestroyShaderEXT(context.device, created_shader, nullptr);
            created_shader = VK_NULL_HANDLE;
        }

        shader_create_info = build_create_info(/*force_spirv=*/true);
        used_binary = false;

        vk_result = vkCreateShadersEXT(context.device, 1, &shader_create_info, nullptr, &created_shader);
    }

    if (vk_result != VK_SUCCESS) {
        vkDestroyPipelineLayout(context.device, result.layout_, nullptr);
        result.layout_ = VK_NULL_HANDLE;
        result.context_ = nullptr;

        return std::unexpected(make_error(
                ShaderObjectErrorType::shader_creation_failed,
                std::format("vkCreateShadersEXT failed for compute shader object '{}'", create_info.debug_name),
                vk_result));
    }

    result.shaders_[0] = created_shader;
    result.stages_[0] = VK_SHADER_STAGE_COMPUTE_BIT;
    result.count_ = 1;
    result.bind_point_ = VK_PIPELINE_BIND_POINT_COMPUTE;

    vk::set_object_name(context.device, VK_OBJECT_TYPE_SHADER_EXT, vk::object_handle(created_shader),
                        create_info.debug_name);
    auto const layout_name = std::string{create_info.debug_name} + ".layout";
    vk::set_object_name(context.device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, vk::object_handle(result.layout_), layout_name);

    if (binary_cache != nullptr && !shader.cache_key.empty() && !used_binary) {
        std::size_t size = 0;
        vkGetShaderBinaryDataEXT(context.device, created_shader, &size, nullptr);

        if (size != 0) {
            std::vector<std::byte> blob(size);

            if (vkGetShaderBinaryDataEXT(context.device, created_shader, &size, blob.data()) == VK_SUCCESS) {
                blob.resize(size);
                binary_cache->store(shader.cache_key, blob);
            }
        }
    }

    return result;
}

auto ShaderObjectSet::bind(VkCommandBuffer command_buffer) const noexcept -> void {
    if (count_ == 0) {
        return;
    }

    if (bind_point_ == VK_PIPELINE_BIND_POINT_GRAPHICS) {
        static constexpr std::array<VkShaderStageFlagBits, 7> conflicting_stages{
                VK_SHADER_STAGE_FRAGMENT_BIT,
                VK_SHADER_STAGE_VERTEX_BIT,
                VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                VK_SHADER_STAGE_GEOMETRY_BIT,
                VK_SHADER_STAGE_TASK_BIT_EXT,
                VK_SHADER_STAGE_MESH_BIT_EXT,
        };

        std::array<VkShaderStageFlagBits, conflicting_stages.size()> stages_to_clear{};
        std::array<VkShaderEXT, conflicting_stages.size()> null_shaders{};
        std::uint32_t clear_count = 0;

        for (auto const stage: conflicting_stages) {
            auto const owned = std::any_of(stages_.begin(), stages_.begin() + count_,
                                           [stage](VkShaderStageFlagBits owned_stage) { return owned_stage == stage; });

            if (!owned) {
                stages_to_clear[clear_count] = stage;
                null_shaders[clear_count] = VK_NULL_HANDLE;
                ++clear_count;
            }
        }

        if (clear_count != 0) {
            vkCmdBindShadersEXT(command_buffer, clear_count, stages_to_clear.data(), null_shaders.data());
        }
    }

    vkCmdBindShadersEXT(command_buffer, count_, stages_.data(), shaders_.data());
}

auto ShaderObjectSet::destroy() noexcept -> void {
    if (context_ == nullptr) {
        return;
    }

    if (context_->device != VK_NULL_HANDLE) {
        for (std::uint32_t index = 0; index < count_; ++index) {
            if (shaders_[index] != VK_NULL_HANDLE) {
                vkDestroyShaderEXT(context_->device, shaders_[index], nullptr);
            }
        }

        if (layout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(context_->device, layout_, nullptr);
        }
    }

    shaders_.fill(VK_NULL_HANDLE);
    stages_.fill(VkShaderStageFlagBits{});
    count_ = 0;

    layout_ = VK_NULL_HANDLE;
    context_ = nullptr;
}
