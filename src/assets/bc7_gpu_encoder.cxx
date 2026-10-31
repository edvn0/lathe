#include "assets/bc7_gpu_encoder.hxx"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <utility>

#include <encoder/basisu_gpu_texture.h>

#include "assets/slang_compiler.hxx"
#include "assets/texture_cache.hxx"
#include "assets/texture_pipeline.hxx"
#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/paths.hxx"
#include "gpu/buffer.hxx"
#include "gpu/context.hxx"
#include "gpu/image.hxx"

namespace {

    struct PushConstants {
        VkDeviceAddress src;
        VkDeviceAddress dst;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t blocks_x;
        std::uint32_t blocks_y;
    };

    constexpr std::uint32_t threads_per_group = 64;

    using Clock = std::chrono::steady_clock;

    auto milliseconds_since(Clock::time_point start) -> double {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    // The image the encoders are given, for measuring how far their output is from it.
    struct SourceImage {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::vector<std::uint8_t> rgba;
    };

    // PSNR of any BC7 image against the original, over RGBA, decoded by the library libktx carries.
    auto psnr_any(std::span<std::byte const> blocks, SourceImage const &original) -> double {
        auto const blocks_x = (original.width + 3) / 4;
        auto const blocks_y = (original.height + 3) / 4;
        auto squared_error = 0.0;
        auto count = 0.0;
        std::array<basisu::color_rgba, 16> decoded{};

        for (std::uint32_t by = 0; by < blocks_y; ++by) {
            for (std::uint32_t bx = 0; bx < blocks_x; ++bx) {
                if (!basisu::unpack_bc7(blocks.data() + (static_cast<std::size_t>(by) * blocks_x + bx) * 16,
                                        decoded.data())) {
                    return -1.0;
                }

                for (std::uint32_t texel = 0; texel < 16; ++texel) {
                    auto const x = bx * 4 + (texel & 3);
                    auto const y = by * 4 + (texel >> 2);

                    if (x >= original.width || y >= original.height) {
                        continue;
                    }

                    for (std::uint32_t channel = 0; channel < 4; ++channel) {
                        auto const difference =
                                static_cast<double>(decoded[texel][channel]) -
                                static_cast<double>(original.rgba[(static_cast<std::size_t>(y) * original.width + x) * 4 + channel]);
                        squared_error += difference * difference;
                        count += 1.0;
                    }
                }
            }
        }

        return squared_error <= 0.0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / (squared_error / count));
    }

}

Bc7GpuEncoder::~Bc7GpuEncoder() { destroy(); }

Bc7GpuEncoder::Bc7GpuEncoder(Bc7GpuEncoder &&other) noexcept :
    context_(std::exchange(other.context_, nullptr)), shader_(std::exchange(other.shader_, VK_NULL_HANDLE)),
    layout_(std::exchange(other.layout_, VK_NULL_HANDLE)), timestamps_(std::exchange(other.timestamps_, VK_NULL_HANDLE)) {}

auto Bc7GpuEncoder::operator=(Bc7GpuEncoder &&other) noexcept -> Bc7GpuEncoder & {
    if (this != &other) {
        destroy();

        context_ = std::exchange(other.context_, nullptr);
        shader_ = std::exchange(other.shader_, VK_NULL_HANDLE);
        layout_ = std::exchange(other.layout_, VK_NULL_HANDLE);
        timestamps_ = std::exchange(other.timestamps_, VK_NULL_HANDLE);
    }

    return *this;
}

auto Bc7GpuEncoder::destroy() noexcept -> void {
    if (context_ == nullptr) {
        return;
    }

    if (shader_ != VK_NULL_HANDLE) {
        vkDestroyShaderEXT(context_->device, shader_, nullptr);
    }

    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context_->device, layout_, nullptr);
    }

    if (timestamps_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(context_->device, timestamps_, nullptr);
    }

    shader_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    timestamps_ = VK_NULL_HANDLE;
    context_ = nullptr;
}

auto Bc7GpuEncoder::create(VulkanContext &context, renderer::SlangCompiler const &compiler)
        -> std::expected<Bc7GpuEncoder, std::string> {
    if (!context.shader_objects_supported) {
        return std::unexpected{"the device has no VK_EXT_shader_object"};
    }

    auto compiled = compiler.compile(renderer::ShaderCompileRequest{
            .source_path = engine_path("assets/shaders/bc7_encode.slang"),
            .entry_point = FlyString{"main_cs"},
            .stage = renderer::ShaderStage::compute,
    });

    if (!compiled) {
        return std::unexpected{std::format("compiling bc7_encode.slang: {}", describe(compiled.error()))};
    }

    Bc7GpuEncoder encoder;
    encoder.context_ = &context;

    VkPushConstantRange const push_range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PushConstants),
    };
    VkPipelineLayoutCreateInfo const layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
    };

    if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &encoder.layout_) != VK_SUCCESS) {
        return std::unexpected{"vkCreatePipelineLayout failed"};
    }

    VkShaderCreateInfoEXT const shader_info{
            .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
            .codeSize = compiled->spirv.size() * sizeof(std::uint32_t),
            .pCode = compiled->spirv.data(),
            .pName = compiled->entry_point.c_str(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
    };

    if (auto const result = vkCreateShadersEXT(context.device, 1, &shader_info, nullptr, &encoder.shader_);
        result != VK_SUCCESS) {
        return std::unexpected{std::format("vkCreateShadersEXT failed (VkResult {})", static_cast<int>(result))};
    }

    VkQueryPoolCreateInfo const query_info{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 2,
    };

    if (vkCreateQueryPool(context.device, &query_info, nullptr, &encoder.timestamps_) != VK_SUCCESS) {
        return std::unexpected{"vkCreateQueryPool failed"};
    }

    return encoder;
}

auto Bc7GpuEncoder::encode(std::span<std::byte const> rgba8, std::uint32_t width, std::uint32_t height)
        -> std::expected<Result, std::string> {
    if (width == 0 || height == 0 || rgba8.size() != static_cast<std::size_t>(width) * height * 4) {
        return std::unexpected{"the image is empty or not RGBA8"};
    }

    auto const start = Clock::now();
    auto const blocks_x = (width + 3) / 4;
    auto const blocks_y = (height + 3) / 4;
    auto const block_count = static_cast<std::size_t>(blocks_x) * blocks_y;
    constexpr auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    auto input = Buffer::create(*context_, BufferCreateInfo{.size = rgba8.size(),
                                                           .usage = usage,
                                                           .memory = BufferMemory::upload,
                                                           .debug_name = "bc7_encoder.input"});
    auto output = Buffer::create(*context_, BufferCreateInfo{.size = block_count * 16,
                                                            .usage = usage,
                                                            .memory = BufferMemory::readback,
                                                            .debug_name = "bc7_encoder.output"});

    if (!input || !output) {
        return std::unexpected{"could not allocate the encoder's buffers"};
    }

    if (!input->write(0, rgba8) || !input->flush()) {
        return std::unexpected{"could not upload the image"};
    }

    PushConstants const constants{
            .src = input->device_address,
            .dst = output->device_address,
            .width = width,
            .height = height,
            .blocks_x = blocks_x,
            .blocks_y = blocks_y,
    };

    context_->one_time_submit([&](VkCommandBuffer commands) {
        vkCmdResetQueryPool(commands, timestamps_, 0, 2);
        vkCmdWriteTimestamp2(commands, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestamps_, 0);

        auto const stage = VK_SHADER_STAGE_COMPUTE_BIT;
        vkCmdBindShadersEXT(commands, 1, &stage, &shader_);
        vkCmdPushConstants(commands, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &constants);
        vkCmdDispatch(commands, static_cast<std::uint32_t>((block_count + threads_per_group - 1) / threads_per_group), 1,
                      1);

        VkMemoryBarrier2 const barrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
        };
        VkDependencyInfo const dependency{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &barrier,
        };
        vkCmdPipelineBarrier2(commands, &dependency);

        vkCmdWriteTimestamp2(commands, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestamps_, 1);
    });

    Result result;
    result.blocks.resize(block_count * 16);

    if (!output->invalidate() || !output->read(0, std::span{result.blocks})) {
        return std::unexpected{"could not read the blocks back"};
    }

    std::array<std::uint64_t, 2> ticks{};

    if (vkGetQueryPoolResults(context_->device, timestamps_, 0, 2, sizeof(ticks), ticks.data(), sizeof(std::uint64_t),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context_->physical_device, &properties);
        result.gpu_ms = static_cast<double>(ticks[1] - ticks[0]) * properties.limits.timestampPeriod * 1e-6;
    }

    result.total_ms = milliseconds_since(start);

    return result;
}

auto run_bc7_benchmark(VulkanContext &context, renderer::SlangCompiler const &compiler,
                       std::filesystem::path const &image_path) -> bool {
    auto decoded = DecodedImage::load_from_file(image_path.string(), ImageColourSpace::linear);

    if (!decoded) {
        error("bc7-bench: could not decode '{}'", image_path.string());

        return false;
    }

    if (decoded->format() != VK_FORMAT_R8G8B8A8_UNORM && decoded->format() != VK_FORMAT_R8G8B8A8_SRGB) {
        error("bc7-bench: '{}' is not an 8-bit RGBA image", image_path.string());

        return false;
    }

    auto const pixels = decoded->span();
    SourceImage original{.width = decoded->width(), .height = decoded->height()};
    original.rgba.resize(pixels.size());
    std::memcpy(original.rgba.data(), pixels.data(), pixels.size());

    info("bc7-bench: {} is {}x{}", image_path.filename().string(), original.width, original.height);

    auto encoder = Bc7GpuEncoder::create(context, compiler);

    if (!encoder) {
        error("bc7-bench: {}", encoder.error());

        return false;
    }

    // The first dispatch pays for pipeline warm-up and driver-side allocations; time the ones after it.
    constexpr auto runs = 5;
    Bc7GpuEncoder::Result best;

    for (auto run = 0; run < runs; ++run) {
        auto result = encoder->encode(pixels, original.width, original.height);

        if (!result) {
            error("bc7-bench: {}", result.error());

            return false;
        }

        info("bc7-bench: GPU run {}: {:.2f} ms on the GPU, {:.2f} ms with copies", run + 1, result->gpu_ms,
             result->total_ms);

        if (run == 0 || result->gpu_ms < best.gpu_ms) {
            best = std::move(*result);
        }
    }

    auto const gpu_psnr = psnr_any(best.blocks, original);

    // The engine's own path for the same image: decode, mips, UASTC, transcode to BC7. A throw-away cache keeps it a
    // cold run, and its time covers every mip level, which is a third more texels than the GPU encoded here.
    auto const cache_dir = std::filesystem::temp_directory_path() /
                           std::format("bc7_bench_{}", Clock::now().time_since_epoch().count());
    auto const cpu_start = Clock::now();
    auto cpu = load_compressed_texture(AssetPath::external(image_path).value(), TextureRole::generic,
                                       TextureCache{cache_dir});
    auto const cpu_ms = milliseconds_since(cpu_start);
    std::error_code ignored;
    std::filesystem::remove_all(cache_dir, ignored);

    info("bc7-bench: GPU best {:.2f} ms on the GPU ({:.2f} ms with copies); PSNR {:.2f} dB", best.gpu_ms,
         best.total_ms, gpu_psnr);

    if (!cpu) {
        warn("bc7-bench: the CPU path failed, nothing to compare with");

        return true;
    }

    // Level 0 of the CPU result, compared the same way if it is BC7 (an encoder that picks other modes will read as
    // mismatches here, which the line below says).
    auto const &base = cpu->mips.front();
    std::span<std::byte const> cpu_blocks{cpu->data.data() + base.byte_offset, base.byte_length};

    std::array<std::size_t, 9> modes{};

    for (std::size_t block = 0; block + 16 <= cpu_blocks.size(); block += 16) {
        auto const low = static_cast<std::uint8_t>(cpu_blocks[block]);
        ++modes[low == 0 ? 8 : static_cast<std::size_t>(std::countr_zero(low))];
    }

    info("bc7-bench: CPU level 0 is {}x{} ({} blocks); blocks per BC7 mode 0..7: {} {} {} {} {} {} {} {}", base.width,
         base.height, cpu_blocks.size() / 16, modes[0], modes[1], modes[2], modes[3], modes[4], modes[5], modes[6],
         modes[7]);
    auto const cpu_psnr = psnr_any(cpu_blocks, original);

    info("bc7-bench: CPU (UASTC to BC7, all levels, one thread's wall time with the pool busy) {:.0f} ms", cpu_ms);

    if (cpu_psnr < 0.0) {
        info("bc7-bench: CPU output uses other BC7 modes than mode 6, so it needs the full decoder for a PSNR");
    } else {
        info("bc7-bench: CPU PSNR {:.2f} dB", cpu_psnr);
    }

    return true;
}
