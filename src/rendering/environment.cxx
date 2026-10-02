#include "rendering/environment.hxx"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>

#include <glm/gtc/packing.hpp>
#include <glm/gtx/component_wise.hpp>
#include <stb_image_resize2.h>
#include <tracy/Tracy.hpp>

#include "core/logger.hxx"
#include "core/thread_pool.hxx"
#include "gpu/context.hxx"
#include "gpu/vk_barrier.hxx"
#include "rendering/brdf_lut.hxx"
#include "rendering/cube_map.hxx"
#include "rendering/sky_model.hxx"
#include "rendering/spherical_harmonics.hxx"

// Generated at build time from the shaders' push_constant blocks (see CMakeLists.txt).
#include "shader_push_constants.hxx"

namespace {

    constexpr VkPipelineStageFlags2 compute_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    constexpr VkPipelineStageFlags2 reader_stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

    // GGX samples for prefilter mips 0-5; mip 0 is a mirror copy and uses none.
    constexpr std::array<std::uint32_t, EnvironmentSystem::prefilter_mips> prefilter_sample_counts{0, 32, 48, 64, 64, 64};

    // The widest equirect uploaded; larger panoramas are downscaled on the CPU first.
    constexpr std::uint32_t max_equirect_width = 8192;

    // Sun disc radiance is clamped so it blooms without flooding the frame.
    constexpr float max_sun_disc_radiance = 2000.0F;

    [[nodiscard]]
    auto make_error(RendererErrorType type) noexcept -> RendererError {
        return RendererError{.type = type};
    }

    [[nodiscard]]
    auto same_handle(ShaderObjectHandle a, ShaderObjectHandle b) noexcept -> bool {
        return a.index == b.index && a.generation == b.generation;
    }

    [[nodiscard]]
    auto dispatch_groups(std::uint32_t size) noexcept -> std::uint32_t {
        return (size + 7U) / 8U;
    }

    template<typename PushConstants>
    auto push(VkCommandBuffer command_buffer, VkPipelineLayout layout, PushConstants const &constants) noexcept -> void {
        vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &constants);
    }

    [[nodiscard]]
    auto quantise(float value, float scale) noexcept -> std::int32_t {
        return static_cast<std::int32_t>(std::lround(value * scale));
    }

    // Below the horizon the sky fades toward a dim night floor between -10 and 0 degrees.
    [[nodiscard]]
    auto night_fade(float elevation_degrees) noexcept -> float {
        constexpr float night_floor = 0.02F;

        auto const t = std::clamp((elevation_degrees + 10.0F) / 10.0F, 0.0F, 1.0F);

        return night_floor + ((1.0F - night_floor) * t * t * (3.0F - (2.0F * t)));
    }

    [[nodiscard]]
    auto sun_direction_of(SceneSun const &sun) noexcept -> glm::vec3 {
        auto const azimuth = glm::radians(sun.azimuth_degrees);
        auto const elevation = glm::radians(sun.elevation_degrees);

        return glm::vec3{std::cos(elevation) * std::cos(azimuth), std::sin(elevation), std::cos(elevation) * std::sin(azimuth)};
    }

} // namespace

EnvironmentSystem::~EnvironmentSystem() { destroy(); }

auto EnvironmentSystem::destroy() noexcept -> void {
    decode_.reset();
    retired_.clear();
    equirect_.reset();
    radiance_ = CubeResource{};
    prefilter_ = {};
    brdf_lut_.reset();
    sh_buffer_.destroy();

    if (samplers_ != nullptr && equirect_sampler_.valid()) {
        static_cast<void>(samplers_->destroy_sampler(equirect_sampler_));
    }

    equirect_sampler_ = {};
    building_.reset();
    live_key_.reset();
    initialised_ = false;
}

auto EnvironmentSystem::create_cube(std::uint32_t size, std::uint32_t mip_count, std::string_view name,
                                    bool transfer_dst) -> std::expected<CubeResource, RendererError> {
    CubeResource cube;
    cube.size = size;
    cube.mip_count = mip_count;

    auto image = create_held_image(
            *images_, ImageCreateInfo{
                              .extent = VkExtent3D{.width = size, .height = size, .depth = 1},
                              .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                              .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT | (transfer_dst ? static_cast<VkImageUsageFlags>(VK_IMAGE_USAGE_TRANSFER_DST_BIT) : 0U),
                              .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                              .image_type = VK_IMAGE_TYPE_2D,
                              .view_type = VK_IMAGE_VIEW_TYPE_CUBE,
                              .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_cube),
                              .flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
                              .samples = VK_SAMPLE_COUNT_1_BIT,
                              .tiling = VK_IMAGE_TILING_OPTIMAL,
                              .mip_levels = mip_count,
                              .array_layers = 6,
                              .create_mip_layer_views = true,
                              .debug_name = name,
                      });

    if (!image) {
        return std::unexpected(make_error(RendererErrorType::image_error));
    }

    cube.image = std::move(*image);

    auto const *raw = cube.image.get();

    cube.face_slots.reserve(static_cast<std::size_t>(mip_count) * 6U);

    for (std::uint32_t mip = 0; mip < mip_count; ++mip) {
        for (std::uint32_t face = 0; face < 6; ++face) {
            auto const view = raw->mip_layer_view(mip, face);

            auto slot = register_held_view(*images_, ImageViewRegistration{.sampled_2d = view, .storage_2d = view});

            if (!slot) {
                return std::unexpected(make_error(RendererErrorType::image_error));
            }

            cube.face_slots.push_back(std::move(*slot));
        }
    }

    return cube;
}

auto EnvironmentSystem::initialize(CreateInfo const &create_info) -> std::expected<void, RendererError> {
    if (create_info.context == nullptr || create_info.images == nullptr || create_info.samplers == nullptr ||
        create_info.pipelines == nullptr) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    context_ = create_info.context;
    images_ = create_info.images;
    samplers_ = create_info.samplers;
    pipelines_ = create_info.pipelines;
    handles_ = create_info.handles;
    frames_in_flight_ = create_info.frames_in_flight;

    auto lut = create_held_image(
            *images_, ImageCreateInfo{
                              .extent = VkExtent3D{.width = brdf_lut_size, .height = brdf_lut_size, .depth = 1},
                              .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                              .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                              .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                              .image_type = VK_IMAGE_TYPE_2D,
                              .view_type = VK_IMAGE_VIEW_TYPE_2D,
                              .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d) |
                                                  image_descriptor_view_bit(ImageDescriptorView::storage_2d),
                              .samples = VK_SAMPLE_COUNT_1_BIT,
                              .tiling = VK_IMAGE_TILING_OPTIMAL,
                              .mip_levels = 1,
                              .array_layers = 1,
                              .debug_name = "environment.brdf_lut",
                      });

    if (!lut) {
        return std::unexpected(make_error(RendererErrorType::image_error));
    }

    brdf_lut_ = std::move(*lut);

    for (std::uint32_t set = 0; set < 2; ++set) {
        auto cube = create_cube(prefilter_size, prefilter_mips, set == 0 ? "environment.prefilter.0" : "environment.prefilter.1", false);

        if (!cube) {
            return std::unexpected(cube.error());
        }

        prefilter_[set] = std::move(*cube);
    }

    auto sh = Buffer::create(*context_, BufferCreateInfo{
                                                .size = 2 * sizeof(GpuEnvironmentSh),
                                                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                .memory = BufferMemory::device,
                                                .debug_name = "environment.sh",
                                        });

    if (!sh) {
        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    sh_buffer_ = std::move(*sh);

    auto sampler = samplers_->create_sampler(SamplerCreateInfo{
            .mag_filter = VK_FILTER_LINEAR,
            .min_filter = VK_FILTER_LINEAR,
            .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .address_mode_u = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .max_anisotropy = 1.0F,
            .min_lod = 0.0F,
            .max_lod = 0.0F,
            .debug_name = "environment.equirect",
    });

    if (!sampler) {
        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    equirect_sampler_ = *sampler;

    std::array<PipelineNodeHandle, 6> const nodes{handles_.brdf_lut,   handles_.equirect_to_cube, handles_.sky_to_cube,
                                                  handles_.downsample, handles_.sh_project,       handles_.prefilter};

    for (std::size_t index = 0; index < nodes.size(); ++index) {
        shader_handles_[index] = pipelines_->shader_object_handle(nodes[index]);
    }

    initialised_ = true;

    return {};
}

auto EnvironmentSystem::retire(CubeResource &&cube) -> void {
    retired_.push_back(Retired{.frame = frame_number_, .cube = std::move(cube), .image = {}, .buffer = {}});
}

auto EnvironmentSystem::retire(ImageHolder &&image) -> void {
    retired_.push_back(Retired{.frame = frame_number_, .cube = {}, .image = std::move(image), .buffer = {}});
}

auto EnvironmentSystem::retire(Buffer &&buffer) -> void {
    retired_.push_back(Retired{.frame = frame_number_, .cube = {}, .image = {}, .buffer = std::move(buffer)});
}

auto EnvironmentSystem::pipelines_ready() const -> bool {
    for (auto const node: {handles_.brdf_lut, handles_.equirect_to_cube, handles_.sky_to_cube, handles_.downsample,
                           handles_.sh_project, handles_.prefilter}) {
        if (pipelines_->resolve_shader_objects(node) == nullptr) {
            return false;
        }
    }

    return true;
}

auto EnvironmentSystem::detect_shader_changes() -> void {
    std::array<PipelineNodeHandle, 6> const nodes{handles_.brdf_lut,   handles_.equirect_to_cube, handles_.sky_to_cube,
                                                  handles_.downsample, handles_.sh_project,       handles_.prefilter};

    for (std::size_t index = 0; index < nodes.size(); ++index) {
        auto const current = pipelines_->shader_object_handle(nodes[index]);

        if (same_handle(current, shader_handles_[index])) {
            continue;
        }

        shader_handles_[index] = current;

        if (index == 0) {
            lut_ready_ = false;
        } else {
            ++generation_;
        }
    }
}

auto EnvironmentSystem::desired_key() const -> BuildKey {
    BuildKey key;
    key.source = desired_.source;
    key.generation = generation_;

    if (desired_.source == EnvironmentSource::hdr_image) {
        key.hdr_source = desired_.hdr_source;
        key.cube_size = desired_.hdr_cube_size;
    } else if (desired_.source == EnvironmentSource::procedural_sky) {
        key.azimuth_centidegrees = quantise(desired_.sun.azimuth_degrees, 100.0F);
        key.elevation_centidegrees = quantise(desired_.sun.elevation_degrees, 100.0F);
        key.turbidity_milli = quantise(desired_.sun.turbidity, 1000.0F);
        key.sky_intensity_milli = quantise(desired_.sky_intensity, 1000.0F);
        key.ground_milli = {quantise(desired_.sun.ground_albedo.x, 1000.0F), quantise(desired_.sun.ground_albedo.y, 1000.0F),
                            quantise(desired_.sun.ground_albedo.z, 1000.0F)};
    }

    return key;
}

auto EnvironmentSystem::ensure_procedural_radiance() -> std::expected<void, RendererError> {
    if (radiance_source_ == EnvironmentSource::procedural_sky && radiance_.image.handle().valid() &&
        radiance_.size == procedural_cube_size) {
        return {};
    }

    if (radiance_.image.handle().valid()) {
        retire(std::move(radiance_));
    }

    auto cube = create_cube(procedural_cube_size, std::bit_width(procedural_cube_size), "environment.radiance", false);

    if (!cube) {
        return std::unexpected(cube.error());
    }

    radiance_ = std::move(*cube);
    radiance_source_ = EnvironmentSource::procedural_sky;
    radiance_hdr_.clear();
    radiance_is_cube_source_ = false;
    radiance_captured_ = false;

    // A build in flight captured into the old cube.
    building_.reset();

    return {};
}

auto EnvironmentSystem::upload_equirect(VkCommandBuffer command_buffer, HdrImage const &image)
        -> std::expected<void, RendererError> {
    auto texture = create_held_image(
            *images_, ImageCreateInfo{
                              .extent = VkExtent3D{.width = image.width, .height = image.height, .depth = 1},
                              .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                              .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                              .image_type = VK_IMAGE_TYPE_2D,
                              .view_type = VK_IMAGE_VIEW_TYPE_2D,
                              .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                              .samples = VK_SAMPLE_COUNT_1_BIT,
                              .tiling = VK_IMAGE_TILING_OPTIMAL,
                              .mip_levels = 1,
                              .array_layers = 1,
                              .debug_name = "environment.equirect",
                      });

    if (!texture) {
        return std::unexpected(make_error(RendererErrorType::image_error));
    }

    auto staging = Buffer::create(*context_, BufferCreateInfo{
                                                     .size = image.pixels.size() * sizeof(std::uint16_t),
                                                     .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                     .memory = BufferMemory::upload,
                                                     .debug_name = "environment.equirect_staging",
                                             });

    if (!staging || !staging->write(0, std::span<std::uint16_t const>{image.pixels})) {
        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    auto const vk_image = texture->get()->image();

    transition_image_layout(command_buffer, vk_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_PIPELINE_STAGE_2_NONE, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_NONE,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

    VkBufferImageCopy2 const region{
            .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
            .imageOffset = {0, 0, 0},
            .imageExtent = {.width = image.width, .height = image.height, .depth = 1},
    };

    VkCopyBufferToImageInfo2 const copy{
            .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
            .srcBuffer = staging->buffer,
            .dstImage = vk_image,
            .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .regionCount = 1,
            .pRegions = &region,
    };

    vkCmdCopyBufferToImage2(command_buffer, &copy);

    transition_image_layout(command_buffer, vk_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT, compute_stage,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

    if (equirect_.handle().valid()) {
        retire(std::move(equirect_));
    }

    equirect_ = std::move(*texture);

    retire(std::move(*staging));

    return {};
}

auto EnvironmentSystem::upload_cube(VkCommandBuffer command_buffer, HdrImage const &image)
        -> std::expected<void, RendererError> {
    auto const face_bytes = static_cast<VkDeviceSize>(image.width) * image.height * 4U * sizeof(std::uint16_t);

    auto staging = Buffer::create(*context_, BufferCreateInfo{
                                                     .size = face_bytes * 6U,
                                                     .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                     .memory = BufferMemory::upload,
                                                     .debug_name = "environment.cube_staging",
                                             });

    if (!staging || !staging->write(0, std::span<std::uint16_t const>{image.pixels})) {
        return std::unexpected(make_error(RendererErrorType::device_error));
    }

    auto const vk_image = radiance_.image->image();

    transition_image_subresources(command_buffer, vk_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_NONE, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_NONE,
                                  VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);

    std::array<VkBufferImageCopy2, 6> regions{};

    for (std::uint32_t face = 0; face < 6; ++face) {
        regions[face] = VkBufferImageCopy2{
                .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                .bufferOffset = face * face_bytes,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = face, .layerCount = 1},
                .imageOffset = {0, 0, 0},
                .imageExtent = {.width = image.width, .height = image.height, .depth = 1},
        };
    }

    VkCopyBufferToImageInfo2 const copy{
            .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
            .srcBuffer = staging->buffer,
            .dstImage = vk_image,
            .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .regionCount = static_cast<std::uint32_t>(regions.size()),
            .pRegions = regions.data(),
    };

    vkCmdCopyBufferToImage2(command_buffer, &copy);

    transition_image_subresources(command_buffer, vk_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT, compute_stage,
                                  VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                  VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);

    retire(std::move(*staging));

    return {};
}

auto EnvironmentSystem::finish_decode(VkCommandBuffer command_buffer, HdrImage const &decoded)
        -> std::expected<void, RendererError> {
    // Panoramas wider than the cap are downscaled before upload; the cube is at most 1024 per face anyway.
    HdrImage const *source = &decoded;
    HdrImage scaled;

    if (decoded.layers == 1 && decoded.width > max_equirect_width) {
        scaled.width = max_equirect_width;
        scaled.height = max_equirect_width / 2;
        scaled.layers = 1;
        scaled.pixels.resize(static_cast<std::size_t>(scaled.width) * scaled.height * 4U);

        if (stbir_resize(decoded.pixels.data(), static_cast<int>(decoded.width), static_cast<int>(decoded.height), 0,
                         scaled.pixels.data(), static_cast<int>(scaled.width), static_cast<int>(scaled.height), 0,
                         STBIR_RGBA, STBIR_TYPE_HALF_FLOAT, STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT) == nullptr) {
            return std::unexpected(make_error(RendererErrorType::image_error));
        }

        source = &scaled;
    }

    auto const is_cube = source->layers == 6;
    auto const size = is_cube ? source->width : std::clamp(std::bit_ceil(desired_.hdr_cube_size), 256U, 1024U);

    if (radiance_.image.handle().valid()) {
        retire(std::move(radiance_));
    }

    auto cube = create_cube(size, std::bit_width(size), "environment.radiance", is_cube);

    if (!cube) {
        return std::unexpected(cube.error());
    }

    radiance_ = std::move(*cube);
    radiance_source_ = EnvironmentSource::hdr_image;
    radiance_hdr_ = decode_ ? decode_->path : std::string{};
    radiance_is_cube_source_ = is_cube;
    radiance_captured_ = false;
    building_.reset();

    if (is_cube) {
        equirect_pending_projection_ = false;

        return upload_cube(command_buffer, *source);
    }

    equirect_pending_projection_ = true;

    return upload_equirect(command_buffer, *source);
}

auto EnvironmentSystem::prepare(VkCommandBuffer command_buffer, std::uint64_t frame_number)
        -> std::expected<void, RendererError> {
    ZoneScopedNC("EnvironmentPrepare", tracy::Color::SkyBlue);

    frame_number_ = frame_number;

    // The GPU is done with anything retired more than a frames-in-flight cycle ago.
    while (!retired_.empty() && retired_.front().frame + frames_in_flight_ + 1 <= frame_number) {
        retired_.erase(retired_.begin());
    }

    plan_ = {};

    if (!initialised_) {
        return {};
    }

    detect_shader_changes();

    plan_.brdf_lut = !lut_ready_ && pipelines_ready();

    if (desired_.source == EnvironmentSource::flat_ambient) {
        building_.reset();
        decode_error_.clear();

        return {};
    }

    if (desired_.source == EnvironmentSource::procedural_sky) {
        if (auto ensured = ensure_procedural_radiance(); !ensured) {
            return ensured;
        }
    } else {
        auto const wanted = desired_.hdr_source;

        auto const loaded_matches = radiance_source_ == EnvironmentSource::hdr_image && radiance_hdr_ == wanted &&
                                    (radiance_is_cube_source_ || radiance_.size == std::clamp(std::bit_ceil(desired_.hdr_cube_size), 256U, 1024U));

        if (!wanted.empty() && !loaded_matches && (!decode_ || decode_->path != wanted) && decode_error_ != wanted) {
            if (auto const provided = provided_.find(wanted); provided != provided_.end()) {
                // Already decoded (a cooked chunk): skip the file, and don't keep a second copy around.
                std::promise<std::expected<HdrImage, HdrImageError>> ready;
                ready.set_value(*provided->second);

                decode_ = DecodeJob{.path = wanted, .future = ready.get_future()};
                provided_.erase(provided);
            } else {
                decode_ = DecodeJob{
                        .path = wanted,
                        .future = thread_pool().submit_task([wanted] {
                            ZoneScopedNC("DecodeEnvironment", tracy::Color::Goldenrod);
                            return load_hdr_image(wanted);
                        }),
                };
            }

            decode_error_.clear();
        }

        if (decode_ && decode_->future.wait_for(std::chrono::seconds{0}) == std::future_status::ready) {
            auto result = decode_->future.get();

            if (result) {
                if (auto finished = finish_decode(command_buffer, *result); !finished) {
                    decode_.reset();
                    return finished;
                }

                loaded_hdr_ = decode_->path;
            } else {
                warn("environment: could not load '{}': {}", decode_->path, result.error().message);

                // Remembered so a failing file is not retried every frame; the path changing clears it.
                decode_error_ = decode_->path;
                error_message_ = result.error().message;
            }

            decode_.reset();
        }
    }

    auto const key = desired_key();

    auto const have_radiance = radiance_.image.handle().valid() && radiance_source_ == desired_.source &&
                               (desired_.source == EnvironmentSource::procedural_sky || radiance_hdr_ == desired_.hdr_source);

    if (!building_ && have_radiance && (!live_key_ || !(*live_key_ == key))) {
        auto const first_build = !live_key_.has_value();

        building_ = Build{
                .key = key,
                .target_set = first_build ? 0U : 1U - live_set_,
                .amortized = debug_.amortize_rebuilds && key.source == EnvironmentSource::procedural_sky && !first_build,
        };
    }

    if (building_) {
        plan_frame(building_->amortized);
    }

    return {};
}

auto EnvironmentSystem::plan_frame(bool /*amortize*/) -> void {
    auto &build = *building_;

    plan_.target_set = build.target_set;

    if (!build.captured) {
        plan_.capture = true;
        build.captured = true;
        radiance_captured_ = true;

        if (!build.amortized) {
            plan_.first_face = 0;
            plan_.face_count = 6;
            build.faces_done = 6;
            plan_.flip = true;
        }
    } else {
        plan_.first_face = build.faces_done;
        plan_.face_count = 1;
        ++build.faces_done;

        plan_.flip = build.faces_done >= 6;
    }

    if (plan_.flip) {
        live_set_ = build.target_set;
        live_key_ = build.key;
        building_.reset();
    }
}

auto EnvironmentSystem::ubo_block() const -> EnvironmentUboBlock {
    EnvironmentUboBlock block;

    auto const &env = desired_;
    auto const live = env.source != EnvironmentSource::flat_ambient && live_key_.has_value() && (lut_ready_ || plan_.brdf_lut);

    block.sampler = samplers_ != nullptr ? samplers_->linear_clamp().index : 0U;
    block.exposure = std::exp2(env.exposure_ev);

    auto const yaw = env.source == EnvironmentSource::hdr_image ? glm::radians(env.rotation_degrees) : 0.0F;

    block.rotation = glm::vec4{std::cos(yaw), std::sin(yaw), static_cast<float>(prefilter_mips - 1), env.specular_occlusion};
    block.intensity = glm::vec4{env.diffuse_intensity, env.specular_intensity, 0.0F, debug_.prefilter_lod};

    auto flags = static_cast<std::uint32_t>(debug_.view) << environment_flag::debug_shift;

    if (brdf_lut_.handle().valid()) {
        block.brdf_lut_texture = brdf_lut_.handle().index;
    }

    if (live) {
        flags |= environment_flag::ibl_valid;
        block.prefilter_cube_texture = prefilter_[live_set_].image.handle().index;
        block.sh_address = sh_buffer_.device_address + (static_cast<VkDeviceSize>(live_set_) * sizeof(GpuEnvironmentSh));

        if (env.specular_occlusion > 0.0F) {
            flags |= environment_flag::specular_occlusion;
        }

        if (env.multi_scatter) {
            flags |= environment_flag::multi_scatter;
        }

        if (env.fog.from_environment) {
            flags |= environment_flag::fog_from_environment;
        }
    }

    if (env.fog_sky) {
        flags |= environment_flag::fog_sky;
    }

    auto const sun_direction = sun_direction_of(env.sun);
    auto const radians = glm::radians(env.sun.elevation_degrees);

    block.sun_direction_cos_radius = glm::vec4{sun_direction, std::cos(glm::radians(env.sun.angular_radius_degrees))};

    if (env.source == EnvironmentSource::procedural_sky) {
        flags |= environment_flag::sky_procedural;

        if (env.draw_skybox) {
            flags |= environment_flag::skybox;
        }

        auto state = make_sky_state({.elevation_radians = radians, .turbidity = env.sun.turbidity});

        // Calibration, the user's scale and the night fade all multiply the luminance channel.
        state.zenith.x *= sky_calibration * env.sky_intensity * night_fade(env.sun.elevation_degrees);

        block.sky_perez = state.perez;
        block.sky_zenith = state.zenith;
        block.sky_ground = glm::vec4{env.sun.ground_albedo, 0.0F};

        auto colour = env.sun.colour;

        if (env.sun.derive_colour_from_sky) {
            colour *= sun_transmittance(radians, env.sun.turbidity);
        }

        auto const disc_solid_angle = 2.0F * glm::pi<float>() * (1.0F - block.sun_direction_cos_radius.w);
        auto const disc_fade = std::clamp((env.sun.elevation_degrees + 2.0F) / 2.0F, 0.0F, 1.0F);

        auto disc = colour * env.sun.intensity * disc_fade / std::max(disc_solid_angle, 1e-7F);

        block.sun_disc_radiance = glm::vec4{glm::min(disc, glm::vec3{max_sun_disc_radiance}), 0.0F};
        block.radiance_cube_texture = radiance_.image.handle().valid() ? radiance_.image.handle().index : 0U;
    } else if (env.source == EnvironmentSource::hdr_image) {
        auto const hdr_ready = radiance_source_ == EnvironmentSource::hdr_image && radiance_hdr_ == env.hdr_source &&
                               radiance_captured_ && radiance_.image.handle().valid();

        if (hdr_ready) {
            if (env.draw_skybox) {
                flags |= environment_flag::skybox;
            }

            block.radiance_cube_texture = radiance_.image.handle().index;
        }
    }

    block.flags = flags;

    return block;
}

auto EnvironmentSystem::status() const -> EnvironmentStatus {
    EnvironmentStatus result;

    result.ibl_live = desired_.source != EnvironmentSource::flat_ambient && live_key_.has_value();

    if (desired_.source == EnvironmentSource::flat_ambient) {
        return result;
    }

    if (!error_message_.empty() && decode_error_ == desired_.hdr_source && desired_.source == EnvironmentSource::hdr_image) {
        result.phase = EnvironmentPhase::failed;
        result.message = error_message_;

        return result;
    }

    if (decode_) {
        result.phase = EnvironmentPhase::decoding;
        result.message = std::format("Decoding {}", decode_->path);

        return result;
    }

    if (building_) {
        result.phase = EnvironmentPhase::building;
        result.step = building_->faces_done;
        result.step_count = 6;

        return result;
    }

    result.phase = live_key_.has_value() && *live_key_ == desired_key() ? EnvironmentPhase::ready : EnvironmentPhase::building;

    return result;
}

auto EnvironmentSystem::brdf_lut_texture_index() const noexcept -> std::uint32_t {
    return brdf_lut_.handle().valid() ? brdf_lut_.handle().index : 0U;
}

auto EnvironmentSystem::radiance_face_texture_index(std::uint32_t mip, std::uint32_t face) const noexcept -> std::uint32_t {
    if (!radiance_.image.handle().valid() || mip >= radiance_.mip_count || face >= 6) {
        return 0;
    }

    return radiance_.face_slot(mip, face).index;
}

auto EnvironmentSystem::prefilter_face_texture_index(std::uint32_t mip, std::uint32_t face) const noexcept -> std::uint32_t {
    auto const &cube = prefilter_[live_set_];

    if (!cube.image.handle().valid() || mip >= cube.mip_count || face >= 6) {
        return 0;
    }

    return cube.face_slot(mip, face).index;
}

auto EnvironmentSystem::live_sh_address() const noexcept -> VkDeviceAddress {
    return sh_buffer_.device_address + (static_cast<VkDeviceSize>(live_set_) * sizeof(GpuEnvironmentSh));
}

auto EnvironmentSystem::record(VkCommandBuffer command_buffer, GpuResourceTable &resource_table,
                               std::uint32_t frame_index, VkDeviceAddress ubo_address) -> void {
    if (!initialised_ || !pipelines_ready()) {
        return;
    }

    auto const bind = [&](PipelineNodeHandle node) -> VkPipelineLayout {
        auto const *objects = pipelines_->resolve_shader_objects(node);

        objects->bind(command_buffer);
        resource_table.bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_COMPUTE, objects->layout());

        return objects->layout();
    };

    auto const sampler = samplers_->linear_clamp().index;

    if (plan_.brdf_lut) {
        auto const image = brdf_lut_->image();

        transition_image_layout(command_buffer, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, reader_stages,
                                compute_stage, VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

        auto const layout = bind(handles_.brdf_lut);

        push(command_buffer, layout,
             EnvBrdfLutPushConstants{
                     .dst_storage_index = brdf_lut_.handle().index,
                     .size = brdf_lut_size,
                     .sample_count = 1024,
             });

        vkCmdDispatch(command_buffer, dispatch_groups(brdf_lut_size), dispatch_groups(brdf_lut_size), 1);

        transition_image_layout(command_buffer, image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                compute_stage, reader_stages, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

        lut_ready_ = true;
    }

    auto const radiance_image = radiance_.image.handle().valid() ? radiance_.image->image() : VK_NULL_HANDLE;
    auto const radiance_index = radiance_.image.handle().index;
    auto const &prefilter = prefilter_[plan_.target_set];

    if (plan_.capture && radiance_image != VK_NULL_HANDLE) {
        // Mip 0: projected from an equirect or rendered from the sky. A cubemap source was uploaded straight into it.
        if (!radiance_is_cube_source_) {
            transition_image_subresources(command_buffer, radiance_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                          reader_stages, compute_stage, VK_ACCESS_2_NONE,
                                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);

            if (radiance_source_ == EnvironmentSource::procedural_sky) {
                auto const layout = bind(handles_.sky_to_cube);

                for (std::uint32_t face = 0; face < 6; ++face) {
                    push(command_buffer, layout,
                         EnvSkyToCubePushConstants{
                                 .ubo_address = ubo_address,
                                 .dst_storage_index = radiance_.face_slot(0, face).index,
                                 .face = face,
                                 .size = radiance_.size,
                         });

                    vkCmdDispatch(command_buffer, dispatch_groups(radiance_.size), dispatch_groups(radiance_.size), 1);
                }
            } else {
                auto const layout = bind(handles_.equirect_to_cube);

                for (std::uint32_t face = 0; face < 6; ++face) {
                    push(command_buffer, layout,
                         EnvEquirectToCubePushConstants{
                                 .equirect_texture_index = equirect_.handle().index,
                                 .sampler_index = equirect_sampler_.index,
                                 .dst_storage_index = radiance_.face_slot(0, face).index,
                                 .face = face,
                                 .size = radiance_.size,
                         });

                    vkCmdDispatch(command_buffer, dispatch_groups(radiance_.size), dispatch_groups(radiance_.size), 1);
                }
            }

            transition_image_subresources(command_buffer, radiance_image, VK_IMAGE_LAYOUT_GENERAL,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, compute_stage, reader_stages,
                                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                          VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);
        }

        if (equirect_pending_projection_ && equirect_.handle().valid()) {
            equirect_pending_projection_ = false;
            retire(std::move(equirect_));
        }

        // The mip chain: each level is a 2x2 box of the one below, read through that face's own single-mip view.
        auto const downsample_layout = bind(handles_.downsample);

        for (std::uint32_t mip = 1; mip < radiance_.mip_count; ++mip) {
            transition_image_subresources(command_buffer, radiance_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                          reader_stages, compute_stage, VK_ACCESS_2_NONE,
                                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 6);

            auto const size = std::max(radiance_.size >> mip, 1U);

            for (std::uint32_t face = 0; face < 6; ++face) {
                push(command_buffer, downsample_layout,
                     EnvDownsamplePushConstants{
                             .src_texture_index = radiance_.face_slot(mip - 1, face).index,
                             .sampler_index = sampler,
                             .dst_storage_index = radiance_.face_slot(mip, face).index,
                             .dst_size = size,
                     });

                vkCmdDispatch(command_buffer, dispatch_groups(size), dispatch_groups(size), 1);
            }

            transition_image_subresources(command_buffer, radiance_image, VK_IMAGE_LAYOUT_GENERAL,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, compute_stage, reader_stages,
                                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                          VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 6);
        }

        // SH into the building slot. The slot is not live, so only earlier readers of its previous contents matter.
        auto const slot_offset = static_cast<VkDeviceSize>(plan_.target_set) * sizeof(GpuEnvironmentSh);

        VkBufferMemoryBarrier2 sh_barrier{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .srcStageMask = reader_stages,
                .srcAccessMask = VK_ACCESS_2_NONE,
                .dstStageMask = compute_stage,
                .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = sh_buffer_.buffer,
                .offset = slot_offset,
                .size = sizeof(GpuEnvironmentSh),
        };

        VkDependencyInfo sh_dependency{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &sh_barrier,
        };

        vkCmdPipelineBarrier2(command_buffer, &sh_dependency);

        auto const sh_level = static_cast<std::uint32_t>(std::max(std::countr_zero(radiance_.size) - 5, 0));

        auto const sh_layout = bind(handles_.sh_project);

        push(command_buffer, sh_layout,
             EnvShProjectPushConstants{
                     .cube_texture_index = radiance_index,
                     .sampler_index = sampler,
                     .level = sh_level,
                     .level_size = radiance_.size >> sh_level,
                     .destination_address = sh_buffer_.device_address + slot_offset,
             });

        vkCmdDispatch(command_buffer, 1, 1, 1);

        sh_barrier.srcStageMask = compute_stage;
        sh_barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        sh_barrier.dstStageMask = reader_stages;
        sh_barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        vkCmdPipelineBarrier2(command_buffer, &sh_dependency);

        // The target prefilter set is rewritten from scratch; the whole image stays GENERAL until it flips.
        transition_image_subresources(command_buffer, prefilter.image->image(), VK_IMAGE_LAYOUT_UNDEFINED,
                                      VK_IMAGE_LAYOUT_GENERAL, reader_stages, compute_stage, VK_ACCESS_2_NONE,
                                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, prefilter_mips, 0, 6);
    }

    if (radiance_image != VK_NULL_HANDLE && (plan_.capture || plan_.face_count != 0)) {
        auto const prefilter_layout = bind(handles_.prefilter);

        auto const dispatch = [&](std::uint32_t mip, std::uint32_t face) {
            auto const size = prefilter_size >> mip;

            push(command_buffer, prefilter_layout,
                 EnvPrefilterPushConstants{
                         .cube_texture_index = radiance_index,
                         .sampler_index = sampler,
                         .dst_storage_index = prefilter.face_slot(mip, face).index,
                         .face = face,
                         .dst_size = size,
                         .sample_count = prefilter_sample_counts[mip],
                         .radiance_size = radiance_.size,
                         .roughness = static_cast<float>(mip) / static_cast<float>(prefilter_mips - 1),
                 });

            vkCmdDispatch(command_buffer, dispatch_groups(size), dispatch_groups(size), 1);
        };

        if (plan_.capture) {
            for (std::uint32_t face = 0; face < 6; ++face) {
                dispatch(0, face);
            }
        }

        for (std::uint32_t face = plan_.first_face; face < plan_.first_face + plan_.face_count; ++face) {
            for (std::uint32_t mip = 1; mip < prefilter_mips; ++mip) {
                dispatch(mip, face);
            }
        }

        if (plan_.flip) {
            transition_image_subresources(command_buffer, prefilter.image->image(), VK_IMAGE_LAYOUT_GENERAL,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, compute_stage, reader_stages,
                                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                          VK_IMAGE_ASPECT_COLOR_BIT, 0, prefilter_mips, 0, 6);
        }
    }
}

auto EnvironmentSystem::validate_against_cpu() -> EnvironmentValidation {
    EnvironmentValidation result;

    if (!initialised_ || !lut_ready_ || !live_key_.has_value() || !radiance_.image.handle().valid() || building_.has_value()) {
        result.summary = "Nothing to validate yet: wait for the environment to finish building.";
        return result;
    }

    auto const sh_level = static_cast<std::uint32_t>(std::max(std::countr_zero(radiance_.size) - 5, 0));
    auto const level_size = radiance_.size >> sh_level;

    auto const lut_bytes = static_cast<VkDeviceSize>(brdf_lut_size) * brdf_lut_size * 4U * sizeof(std::uint16_t);
    auto const face_bytes = static_cast<VkDeviceSize>(level_size) * level_size * 4U * sizeof(std::uint16_t);

    auto const make_readback = [&](VkDeviceSize size, char const *name) {
        return Buffer::create(*context_, BufferCreateInfo{
                                                 .size = size,
                                                 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                 .memory = BufferMemory::readback,
                                                 .debug_name = name,
                                         });
    };

    auto lut_readback = make_readback(lut_bytes, "environment.validate_lut");
    auto radiance_readback = make_readback(face_bytes * 6U, "environment.validate_radiance");
    auto sh_readback = make_readback(sizeof(GpuEnvironmentSh), "environment.validate_sh");

    if (!lut_readback || !radiance_readback || !sh_readback) {
        result.summary = "Could not allocate readback buffers.";
        return result;
    }

    auto const lut_image = brdf_lut_->image();
    auto const radiance_image = radiance_.image->image();
    auto const sh_offset = static_cast<VkDeviceSize>(live_set_) * sizeof(GpuEnvironmentSh);

    context_->one_time_submit([&](VkCommandBuffer command_buffer) {
        // The environment is in sampled layouts, written by earlier frames: only their writes need to be visible.
        transition_image_layout(command_buffer, lut_image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, reader_stages | compute_stage,
                                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_ACCESS_2_TRANSFER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

        VkBufferImageCopy2 const lut_region{
                .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                .imageExtent = {.width = brdf_lut_size, .height = brdf_lut_size, .depth = 1},
        };

        VkCopyImageToBufferInfo2 const lut_copy{
                .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
                .srcImage = lut_image,
                .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .dstBuffer = lut_readback->buffer,
                .regionCount = 1,
                .pRegions = &lut_region,
        };

        vkCmdCopyImageToBuffer2(command_buffer, &lut_copy);

        transition_image_layout(command_buffer, lut_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT, reader_stages,
                                VK_ACCESS_2_TRANSFER_READ_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

        transition_image_subresources(command_buffer, radiance_image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, reader_stages | compute_stage,
                                      VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                      VK_ACCESS_2_TRANSFER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT, sh_level, 1, 0, 6);

        std::array<VkBufferImageCopy2, 6> regions{};

        for (std::uint32_t face = 0; face < 6; ++face) {
            regions[face] = VkBufferImageCopy2{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                    .bufferOffset = face * face_bytes,
                    .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = sh_level, .baseArrayLayer = face, .layerCount = 1},
                    .imageExtent = {.width = level_size, .height = level_size, .depth = 1},
            };
        }

        VkCopyImageToBufferInfo2 const radiance_copy{
                .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
                .srcImage = radiance_image,
                .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .dstBuffer = radiance_readback->buffer,
                .regionCount = static_cast<std::uint32_t>(regions.size()),
                .pRegions = regions.data(),
        };

        vkCmdCopyImageToBuffer2(command_buffer, &radiance_copy);

        transition_image_subresources(command_buffer, radiance_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT, reader_stages,
                                      VK_ACCESS_2_TRANSFER_READ_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                      VK_IMAGE_ASPECT_COLOR_BIT, sh_level, 1, 0, 6);

        VkBufferMemoryBarrier2 const sh_barrier{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .srcStageMask = compute_stage,
                .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = sh_buffer_.buffer,
                .offset = sh_offset,
                .size = sizeof(GpuEnvironmentSh),
        };

        VkDependencyInfo const sh_dependency{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &sh_barrier,
        };

        vkCmdPipelineBarrier2(command_buffer, &sh_dependency);

        VkBufferCopy const sh_region{.srcOffset = sh_offset, .dstOffset = 0, .size = sizeof(GpuEnvironmentSh)};

        vkCmdCopyBuffer(command_buffer, sh_buffer_.buffer, sh_readback->buffer, 1, &sh_region);
    });

    std::vector<std::uint16_t> lut_halves(static_cast<std::size_t>(brdf_lut_size) * brdf_lut_size * 4U);
    std::vector<std::uint16_t> radiance_halves(static_cast<std::size_t>(level_size) * level_size * 6U * 4U);
    GpuEnvironmentSh gpu_sh{};

    if (!lut_readback->read(0, std::span{lut_halves}) || !radiance_readback->read(0, std::span{radiance_halves}) ||
        !sh_readback->read(0, std::span{&gpu_sh, 1})) {
        result.summary = "Could not read the results back.";
        return result;
    }

    // The LUT at a 4x4 grid of texels, against the CPU integral with the same sample set.
    for (std::uint32_t gy = 0; gy < 4; ++gy) {
        for (std::uint32_t gx = 0; gx < 4; ++gx) {
            auto const x = (gx * 32U) + 16U;
            auto const y = (gy * 32U) + 16U;

            auto const reference =
                    integrate_brdf(std::max((static_cast<float>(x) + 0.5F) / static_cast<float>(brdf_lut_size), 1e-3F),
                                   std::max((static_cast<float>(y) + 0.5F) / static_cast<float>(brdf_lut_size), 0.045F), 1024);

            auto const texel = ((static_cast<std::size_t>(y) * brdf_lut_size) + x) * 4U;
            glm::vec2 const gpu{glm::unpackHalf1x16(lut_halves[texel]), glm::unpackHalf1x16(lut_halves[texel + 1])};

            result.lut_max_error = std::max(result.lut_max_error, glm::compMax(glm::abs(gpu - reference)));
        }
    }

    // The SH of the same level, from the same texels.
    Sh9 cpu_sh;

    for (std::uint32_t face = 0; face < 6; ++face) {
        for (std::uint32_t y = 0; y < level_size; ++y) {
            for (std::uint32_t x = 0; x < level_size; ++x) {
                auto const uv = glm::vec2{(static_cast<float>(x) + 0.5F) / static_cast<float>(level_size),
                                          (static_cast<float>(y) + 0.5F) / static_cast<float>(level_size)};

                auto const texel = ((((static_cast<std::size_t>(face) * level_size) + y) * level_size) + x) * 4U;
                glm::vec3 const radiance{glm::unpackHalf1x16(radiance_halves[texel]), glm::unpackHalf1x16(radiance_halves[texel + 1]),
                                         glm::unpackHalf1x16(radiance_halves[texel + 2])};

                sh9_accumulate(cpu_sh, cube_texel_direction(face, uv), radiance, cube_texel_solid_angle(x, y, level_size));
            }
        }
    }

    cpu_sh = sh9_cosine_convolve_over_pi(cpu_sh);

    float largest = 1e-6F;
    float worst = 0.0F;

    for (std::uint32_t index = 0; index < sh9_coefficient_count; ++index) {
        glm::vec3 const gpu{gpu_sh.coefficients[index]};

        largest = std::max(largest, glm::compMax(glm::abs(cpu_sh.coefficients[index])));
        worst = std::max(worst, glm::compMax(glm::abs(gpu - cpu_sh.coefficients[index])));
    }

    result.sh_max_relative_error = worst / largest;

    result.ran = true;
    result.passed = result.lut_max_error < 1e-2F && result.sh_max_relative_error < 1e-3F;
    result.summary = std::format("{}: LUT max error {:.5f} (limit 0.01), SH max relative error {:.6f} (limit 0.001)",
                                 result.passed ? "PASS" : "FAIL", result.lut_max_error, result.sh_max_relative_error);

    info("environment: {}", result.summary);

    return result;
}
