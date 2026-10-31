#include "assets/gpu_texture_encoder.hxx"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <limits>
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
        VkDeviceAddress jobs;
        std::uint32_t job_count;
        std::uint32_t total_blocks;
        std::uint32_t format;
    };

    constexpr std::uint32_t threads_per_group = 64;
    constexpr std::uint64_t submit_timeout_ns = 30'000'000'000ULL;
    constexpr auto buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

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

    // PSNR of a BC7 or BC5 image against the original, decoded by the library libktx carries. BC7 is scored over RGBA,
    // BC5 over the two channels it keeps (red and green).
    auto psnr_of(std::span<std::byte const> blocks, SourceImage const &original, GpuBlockFormat format) -> double {
        auto const blocks_x = (original.width + 3) / 4;
        auto const blocks_y = (original.height + 3) / 4;
        auto const channels = format == GpuBlockFormat::bc5 ? 2U : 4U;
        auto squared_error = 0.0;
        auto count = 0.0;
        std::array<basisu::color_rgba, 16> decoded{};

        for (std::uint32_t by = 0; by < blocks_y; ++by) {
            for (std::uint32_t bx = 0; bx < blocks_x; ++bx) {
                auto const *block = blocks.data() + (static_cast<std::size_t>(by) * blocks_x + bx) * 16;

                if (!basisu::unpack_block(format == GpuBlockFormat::bc5 ? basisu::texture_format::cBC5
                                                                        : basisu::texture_format::cBC7,
                                          block, decoded.data())) {
                    return -1.0;
                }

                for (std::uint32_t texel = 0; texel < 16; ++texel) {
                    auto const x = bx * 4 + (texel & 3);
                    auto const y = by * 4 + (texel >> 2);

                    if (x >= original.width || y >= original.height) {
                        continue;
                    }

                    for (std::uint32_t channel = 0; channel < channels; ++channel) {
                        auto const difference =
                                static_cast<double>(decoded[texel][channel]) -
                                static_cast<double>(
                                        original.rgba[(static_cast<std::size_t>(y) * original.width + x) * 4 +
                                                      channel]);
                        squared_error += difference * difference;
                        count += 1.0;
                    }
                }
            }
        }

        return squared_error <= 0.0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / (squared_error / count));
    }

    auto psnr_any(std::span<std::byte const> blocks, SourceImage const &original) -> double {
        return psnr_of(blocks, original, GpuBlockFormat::bc7);
    }

    // Makes sure 'buffer' holds at least 'size' bytes, growing it (and dropping the old contents) when it does not.
    auto ensure_capacity(VulkanContext &context, Buffer &buffer, VkDeviceSize size, BufferMemory memory,
                         std::string_view name) -> bool {
        if (buffer.valid() && buffer.size() >= size) {
            return true;
        }

        auto created = Buffer::create(context, BufferCreateInfo{.size = std::max<VkDeviceSize>(size, size + size / 2),
                                                                .usage = buffer_usage,
                                                                .memory = memory,
                                                                .debug_name = name});

        if (!created) {
            return false;
        }

        buffer = std::move(*created);

        return true;
    }

} // namespace

struct GpuTextureEncoder::Impl {
    VulkanContext *context = nullptr;
    VkShaderEXT shader = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkQueryPool timestamps = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    // Kept between calls and grown as needed, so a steady stream of batches allocates nothing.
    Buffer input;
    Buffer jobs;
    Buffer output;

    ~Impl() {
        if (context == nullptr) {
            return;
        }

        // The buffers go first; they only need the allocator.
        input = Buffer{};
        jobs = Buffer{};
        output = Buffer{};

        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(context->device, fence, nullptr);
        }

        if (pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(context->device, pool, nullptr);
        }

        if (timestamps != VK_NULL_HANDLE) {
            vkDestroyQueryPool(context->device, timestamps, nullptr);
        }

        if (shader != VK_NULL_HANDLE) {
            vkDestroyShaderEXT(context->device, shader, nullptr);
        }

        if (layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(context->device, layout, nullptr);
        }
    }
};

GpuTextureEncoder::GpuTextureEncoder() = default;
GpuTextureEncoder::~GpuTextureEncoder() = default;
GpuTextureEncoder::GpuTextureEncoder(GpuTextureEncoder &&other) noexcept = default;
auto GpuTextureEncoder::operator=(GpuTextureEncoder &&other) noexcept -> GpuTextureEncoder & = default;

auto GpuTextureEncoder::create(VulkanContext &context, renderer::SlangCompiler const &compiler)
        -> std::expected<GpuTextureEncoder, std::string> {
    if (!context.shader_objects_supported) {
        return std::unexpected{"the device has no VK_EXT_shader_object"};
    }

    if (context.background_queue == VK_NULL_HANDLE) {
        return std::unexpected{"the device has no spare compute queue for background work"};
    }

    auto compiled = compiler.compile(renderer::ShaderCompileRequest{
            .source_path = engine_path("assets/shaders/texture_compress.slang"),
            .entry_point = FlyString{"main_cs"},
            .stage = renderer::ShaderStage::compute,
    });

    if (!compiled) {
        return std::unexpected{std::format("compiling texture_compress.slang: {}", describe(compiled.error()))};
    }

    GpuTextureEncoder encoder;
    encoder.impl_ = std::make_unique<Impl>();
    auto &impl = *encoder.impl_;
    impl.context = &context;

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

    if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &impl.layout) != VK_SUCCESS) {
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

    if (auto const result = vkCreateShadersEXT(context.device, 1, &shader_info, nullptr, &impl.shader);
        result != VK_SUCCESS) {
        return std::unexpected{std::format("vkCreateShadersEXT failed (VkResult {})", static_cast<int>(result))};
    }

    VkQueryPoolCreateInfo const query_info{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 2,
    };

    VkCommandPoolCreateInfo const pool_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = context.queue_families.compute,
    };

    VkFenceCreateInfo const fence_info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};

    if (vkCreateQueryPool(context.device, &query_info, nullptr, &impl.timestamps) != VK_SUCCESS ||
        vkCreateCommandPool(context.device, &pool_info, nullptr, &impl.pool) != VK_SUCCESS ||
        vkCreateFence(context.device, &fence_info, nullptr, &impl.fence) != VK_SUCCESS) {
        return std::unexpected{"could not create the encoder's query pool, command pool or fence"};
    }

    VkCommandBufferAllocateInfo const allocate_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = impl.pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
    };

    if (vkAllocateCommandBuffers(context.device, &allocate_info, &impl.commands) != VK_SUCCESS) {
        return std::unexpected{"vkAllocateCommandBuffers failed"};
    }

    return encoder;
}

auto GpuTextureEncoder::encode_batch(std::span<GpuTextureLevel const> levels, GpuBlockFormat format)
        -> std::expected<Result, std::string> {
    if (impl_ == nullptr) {
        return std::unexpected{"the encoder was not created"};
    }

    if (levels.empty()) {
        return Result{};
    }

    auto &impl = *impl_;
    auto &context = *impl.context;
    auto const start = Clock::now();

    // Lay the levels out one after another in the input, and number their blocks the same way.
    std::vector<std::uint32_t> job_words;
    job_words.reserve(levels.size() * 8);
    std::vector<std::size_t> block_starts;
    block_starts.reserve(levels.size() + 1);
    std::uint64_t texels = 0;
    std::size_t blocks = 0;

    for (auto const &level: levels) {
        if (level.width == 0 || level.height == 0 ||
            level.rgba.size() != static_cast<std::size_t>(level.width) * level.height * 4) {
            return std::unexpected{"a level is empty or not RGBA8"};
        }

        block_starts.push_back(blocks);
        job_words.insert(job_words.end(), {static_cast<std::uint32_t>(texels), level.width, level.height,
                                           (level.width + 3) / 4, static_cast<std::uint32_t>(blocks), 0, 0, 0});
        texels += static_cast<std::uint64_t>(level.width) * level.height;
        blocks += block_count_4x4(level.width, level.height);
    }

    block_starts.push_back(blocks);

    if (texels > std::numeric_limits<std::uint32_t>::max() || blocks > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected{"the batch is too large for 32-bit indexing; split it"};
    }

    if (!ensure_capacity(context, impl.input, texels * 4, BufferMemory::upload, "gpu_texture_encoder.input") ||
        !ensure_capacity(context, impl.jobs, job_words.size() * sizeof(std::uint32_t), BufferMemory::upload,
                         "gpu_texture_encoder.jobs") ||
        !ensure_capacity(context, impl.output, blocks * 16, BufferMemory::readback, "gpu_texture_encoder.output")) {
        return std::unexpected{"could not allocate the encoder's buffers"};
    }

    auto *destination = impl.input.mapped_data();

    for (auto const &level: levels) {
        std::memcpy(destination, level.rgba.data(), level.rgba.size());
        destination += level.rgba.size();
    }

    std::memcpy(impl.jobs.mapped_data(), job_words.data(), job_words.size() * sizeof(std::uint32_t));

    if (!impl.input.flush() || !impl.jobs.flush()) {
        return std::unexpected{"could not upload the batch"};
    }

    PushConstants const constants{
            .src = impl.input.device_address,
            .dst = impl.output.device_address,
            .jobs = impl.jobs.device_address,
            .job_count = static_cast<std::uint32_t>(levels.size()),
            .total_blocks = static_cast<std::uint32_t>(blocks),
            .format = static_cast<std::uint32_t>(format),
    };

    vkResetCommandBuffer(impl.commands, 0);

    VkCommandBufferBeginInfo const begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(impl.commands, &begin_info);

    vkCmdResetQueryPool(impl.commands, impl.timestamps, 0, 2);
    vkCmdWriteTimestamp2(impl.commands, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, impl.timestamps, 0);

    auto const stage = VK_SHADER_STAGE_COMPUTE_BIT;
    vkCmdBindShadersEXT(impl.commands, 1, &stage, &impl.shader);
    vkCmdPushConstants(impl.commands, impl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &constants);
    vkCmdDispatch(impl.commands, static_cast<std::uint32_t>((blocks + threads_per_group - 1) / threads_per_group), 1,
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
    vkCmdPipelineBarrier2(impl.commands, &dependency);

    vkCmdWriteTimestamp2(impl.commands, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, impl.timestamps, 1);
    vkEndCommandBuffer(impl.commands);

    VkCommandBufferSubmitInfo const command_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .commandBuffer = impl.commands,
    };
    VkSubmitInfo2 const submit_info{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &command_info,
    };

    vkResetFences(context.device, 1, &impl.fence);

    if (auto const submitted = vkQueueSubmit2(context.background_queue, 1, &submit_info, impl.fence);
        submitted != VK_SUCCESS) {
        return std::unexpected{std::format("vkQueueSubmit2 failed (VkResult {})", static_cast<int>(submitted))};
    }

    if (auto const waited = vkWaitForFences(context.device, 1, &impl.fence, VK_TRUE, submit_timeout_ns);
        waited != VK_SUCCESS) {
        // The GPU may still be using the buffers, so the encoder is not safe to use again.
        return std::unexpected{std::format("waiting for the encoder failed (VkResult {})", static_cast<int>(waited))};
    }

    if (!impl.output.invalidate()) {
        return std::unexpected{"could not read the blocks back"};
    }

    Result result;
    result.blocks.reserve(levels.size());

    for (std::size_t index = 0; index < levels.size(); ++index) {
        auto const *begin = impl.output.mapped_data() + block_starts[index] * 16;
        result.blocks.emplace_back(begin, begin + (block_starts[index + 1] - block_starts[index]) * 16);
    }

    std::array<std::uint64_t, 2> ticks{};

    if (vkGetQueryPoolResults(context.device, impl.timestamps, 0, 2, sizeof(ticks), ticks.data(), sizeof(std::uint64_t),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context.physical_device, &properties);
        result.gpu_ms = static_cast<double>(ticks[1] - ticks[0]) * properties.limits.timestampPeriod * 1e-6;
    }

    result.total_ms = milliseconds_since(start);

    return result;
}

auto GpuTextureEncoder::encode(std::span<std::byte const> rgba8, std::uint32_t width, std::uint32_t height,
                               GpuBlockFormat format) -> std::expected<Result, std::string> {
    auto const level = GpuTextureLevel{.rgba = rgba8, .width = width, .height = height};

    return encode_batch(std::span{&level, 1}, format);
}

namespace {

    // Decodes 'path' to 8-bit RGBA, or says why not.
    auto load_rgba(std::filesystem::path const &path) -> std::optional<SourceImage> {
        auto decoded = DecodedImage::load_from_file(path.string(), ImageColourSpace::linear);

        if (!decoded) {
            warn("gpu-texture-bench: could not decode '{}'", path.string());

            return std::nullopt;
        }

        if (decoded->format() != VK_FORMAT_R8G8B8A8_UNORM && decoded->format() != VK_FORMAT_R8G8B8A8_SRGB) {
            warn("gpu-texture-bench: '{}' is not an 8-bit RGBA image", path.string());

            return std::nullopt;
        }

        auto const pixels = decoded->span();
        SourceImage image{.width = decoded->width(), .height = decoded->height()};
        image.rgba.resize(pixels.size());
        std::memcpy(image.rgba.data(), pixels.data(), pixels.size());

        return image;
    }

    auto as_level(SourceImage const &image) -> GpuTextureLevel {
        return GpuTextureLevel{
                .rgba = std::as_bytes(std::span{image.rgba}), .width = image.width, .height = image.height};
    }

    // One image, GPU against the engine's CPU path.
    auto benchmark_image(GpuTextureEncoder &encoder, std::filesystem::path const &image_path) -> bool {
        auto const loaded = load_rgba(image_path);

        if (!loaded) {
            return false;
        }

        auto const &original = *loaded;
        info("gpu-texture-bench: {} is {}x{}", image_path.filename().string(), original.width, original.height);

        // The first dispatch pays for warm-up and driver-side allocations; time the ones after it.
        constexpr auto runs = 5;
        GpuTextureEncoder::Result best;

        for (auto run = 0; run < runs; ++run) {
            auto result = encoder.encode(std::as_bytes(std::span{original.rgba}), original.width, original.height);

            if (!result) {
                error("gpu-texture-bench: {}", result.error());

                return false;
            }

            info("gpu-texture-bench: GPU run {}: {:.2f} ms on the GPU, {:.2f} ms with copies", run + 1, result->gpu_ms,
                 result->total_ms);

            if (run == 0 || result->gpu_ms < best.gpu_ms) {
                best = std::move(*result);
            }
        }

        auto const gpu_psnr = psnr_any(best.blocks.front(), original);

        // The engine's own path for the same image: decode, mips, UASTC, transcode to BC7. A throw-away cache keeps it
        // a cold run, and its time covers every mip level, which is a third more texels than the GPU encoded here.
        auto const cache_dir = std::filesystem::temp_directory_path() /
                               std::format("gpu_texture_bench_{}", Clock::now().time_since_epoch().count());
        auto const cpu_start = Clock::now();
        auto cpu = load_compressed_texture(AssetPath::external(image_path).value(), TextureRole::generic,
                                           TextureCache{cache_dir});
        auto const cpu_ms = milliseconds_since(cpu_start);
        std::error_code ignored;
        std::filesystem::remove_all(cache_dir, ignored);

        info("gpu-texture-bench: GPU best {:.2f} ms on the GPU ({:.2f} ms with copies); PSNR {:.2f} dB", best.gpu_ms,
             best.total_ms, gpu_psnr);

        if (!cpu) {
            warn("gpu-texture-bench: the CPU path failed, nothing to compare with");

            return true;
        }

        auto const &base = cpu->mips.front();
        std::span<std::byte const> cpu_blocks{cpu->data.data() + base.byte_offset, base.byte_length};

        info("gpu-texture-bench: CPU (UASTC to BC7, all levels, one thread's wall time with the pool busy) {:.0f} ms",
             cpu_ms);
        info("gpu-texture-bench: CPU PSNR {:.2f} dB", psnr_any(cpu_blocks, original));

        // The same image as a normal map: BC5 on the GPU against the engine's CPU path for normal maps.
        auto bc5 = encoder.encode(std::as_bytes(std::span{original.rgba}), original.width, original.height,
                                  GpuBlockFormat::bc5);

        if (!bc5) {
            error("gpu-texture-bench: {}", bc5.error());

            return false;
        }

        auto bc5_best = std::move(*bc5);

        for (auto run = 1; run < runs; ++run) {
            if (auto again = encoder.encode(std::as_bytes(std::span{original.rgba}), original.width, original.height,
                                            GpuBlockFormat::bc5);
                again && again->gpu_ms < bc5_best.gpu_ms) {
                bc5_best = std::move(*again);
            }
        }

        info("gpu-texture-bench: BC5 on the GPU {:.2f} ms ({:.2f} ms with copies); PSNR over R and G {:.2f} dB",
             bc5_best.gpu_ms, bc5_best.total_ms, psnr_of(bc5_best.blocks.front(), original, GpuBlockFormat::bc5));

        auto const normal_cache_dir = std::filesystem::temp_directory_path() /
                                      std::format("gpu_texture_bench_n_{}", Clock::now().time_since_epoch().count());
        auto cpu_normal = load_compressed_texture(AssetPath::external(image_path).value(), TextureRole::normal_map,
                                                  TextureCache{normal_cache_dir});
        std::filesystem::remove_all(normal_cache_dir, ignored);

        if (cpu_normal) {
            auto const &normal_base = cpu_normal->mips.front();
            std::span<std::byte const> normal_blocks{cpu_normal->data.data() + normal_base.byte_offset,
                                                     normal_base.byte_length};

            info("gpu-texture-bench: BC5 on the CPU PSNR over R and G {:.2f} dB",
                 psnr_of(normal_blocks, original, GpuBlockFormat::bc5));
        }

        return true;
    }

    // Every image of a directory: all at once in batches, then one at a time.
    auto benchmark_directory(GpuTextureEncoder &encoder, std::filesystem::path const &directory) -> bool {
        std::vector<std::filesystem::path> files;

        for (auto const &entry: std::filesystem::directory_iterator{directory}) {
            auto extension = entry.path().extension().string();
            std::ranges::transform(extension, extension.begin(), [](unsigned char c) { return std::tolower(c); });

            if (entry.is_regular_file() && (extension == ".png" || extension == ".jpg" || extension == ".jpeg")) {
                files.push_back(entry.path());
            }
        }

        std::ranges::sort(files);

        if (files.empty()) {
            error("gpu-texture-bench: no PNG or JPEG files in '{}'", directory.string());

            return false;
        }

        info("gpu-texture-bench: decoding {} images from '{}'", files.size(), directory.string());

        auto const decode_start = Clock::now();
        std::vector<SourceImage> images;
        std::vector<std::string> names;

        for (auto index = std::size_t{0}; index < files.size(); ++index) {
            if (auto image = load_rgba(files[index])) {
                images.push_back(std::move(*image));
                names.push_back(files[index].filename().string());
            }

            if ((index + 1) % 10 == 0 || index + 1 == files.size()) {
                info("gpu-texture-bench: decoded {}/{}", index + 1, files.size());
            }
        }

        auto const decode_ms = milliseconds_since(decode_start);
        auto texels = std::uint64_t{0};

        for (auto const &image: images) {
            texels += static_cast<std::uint64_t>(image.width) * image.height;
        }

        info("gpu-texture-bench: {} images, {:.1f} Mtexel, decoded in {:.0f} ms", images.size(),
             static_cast<double>(texels) * 1e-6, decode_ms);

        // Batches of at most this many bytes of pixels, so a big directory does not need all of it on the GPU at once.
        constexpr std::uint64_t batch_bytes = 256ULL * 1024 * 1024;
        std::vector<std::vector<std::byte>> batched(images.size());
        auto batched_gpu_ms = 0.0;
        auto batched_total_ms = 0.0;
        auto batch_count = 0;

        for (auto first = std::size_t{0}; first < images.size();) {
            std::vector<GpuTextureLevel> levels;
            auto bytes = std::uint64_t{0};
            auto last = first;

            while (last < images.size() && (levels.empty() || bytes + images[last].rgba.size() <= batch_bytes)) {
                levels.push_back(as_level(images[last]));
                bytes += images[last].rgba.size();
                ++last;
            }

            // Once to warm up, then the one that counts.
            auto result = encoder.encode_batch(levels);

            if (result) {
                result = encoder.encode_batch(levels);
            }

            if (!result) {
                error("gpu-texture-bench: {}", result.error());

                return false;
            }

            batched_gpu_ms += result->gpu_ms;
            batched_total_ms += result->total_ms;
            ++batch_count;

            for (auto index = first; index < last; ++index) {
                batched[index] = std::move(result->blocks[index - first]);
            }

            info("gpu-texture-bench: batch {} ({} images, {:.0f} MB): {:.2f} ms on the GPU, {:.2f} ms with copies",
                 batch_count, last - first, static_cast<double>(bytes) / 1e6, result->gpu_ms, result->total_ms);

            first = last;
        }

        auto single_gpu_ms = 0.0;
        auto single_total_ms = 0.0;

        for (auto const &image: images) {
            auto result = encoder.encode(std::as_bytes(std::span{image.rgba}), image.width, image.height);

            if (!result) {
                error("gpu-texture-bench: {}", result.error());

                return false;
            }

            single_gpu_ms += result->gpu_ms;
            single_total_ms += result->total_ms;
        }

        auto psnr_sum = 0.0;
        auto psnr_min = 1e9;
        auto worst = std::size_t{0};

        for (auto index = std::size_t{0}; index < images.size(); ++index) {
            auto const psnr = std::min(psnr_any(batched[index], images[index]), 99.0);
            psnr_sum += psnr;

            if (psnr < psnr_min) {
                psnr_min = psnr;
                worst = index;
            }
        }

        info("gpu-texture-bench: batched in {} dispatch(es): {:.2f} ms on the GPU, {:.2f} ms with copies", batch_count,
             batched_gpu_ms, batched_total_ms);
        info("gpu-texture-bench: one at a time ({} dispatches): {:.2f} ms on the GPU, {:.2f} ms with copies",
             images.size(), single_gpu_ms, single_total_ms);
        info("gpu-texture-bench: batching is {:.1f}x faster end to end",
             single_total_ms / std::max(batched_total_ms, 1e-6));
        info("gpu-texture-bench: PSNR mean {:.2f} dB, lowest {:.2f} dB ({})",
             psnr_sum / static_cast<double>(images.size()), psnr_min, names[worst]);

        return true;
    }

} // namespace

auto run_gpu_texture_benchmark(VulkanContext &context, renderer::SlangCompiler const &compiler,
                               std::filesystem::path const &path) -> bool {
    auto encoder = GpuTextureEncoder::create(context, compiler);

    if (!encoder) {
        error("gpu-texture-bench: {}", encoder.error());

        return false;
    }

    return std::filesystem::is_directory(path) ? benchmark_directory(*encoder, path) : benchmark_image(*encoder, path);
}
