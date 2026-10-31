#pragma once

#include <volk.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/forward.hxx"

namespace renderer {
    class SlangCompiler;
}

// What the GPU writes. Both are 16 bytes per 4x4 block.
enum class GpuBlockFormat : std::uint8_t {
    // Each block takes the better of mode 6 (RGBA together, 7 bits per channel: gradients) and mode 5 (alpha apart from
    // colour, 8-bit alpha: the hard edges of cut-outs).
    bc7,
    // The red and green channels as two BC4 blocks, for tangent-space normal maps (the other channels are dropped).
    bc5,
};

// One image, or one mip level, to compress: 8-bit RGBA, rows one after another.
struct GpuTextureLevel {
    std::span<std::byte const> rgba;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

// Number of 16-byte BC7 blocks an image of this size compresses to.
[[nodiscard]] constexpr auto block_count_4x4(std::uint32_t width, std::uint32_t height) noexcept -> std::size_t {
    return static_cast<std::size_t>((width + 3) / 4) * ((height + 3) / 4);
}

// Compresses RGBA8 images to BC7 or BC5 on the GPU, a whole batch in one dispatch: one compute thread per 4x4 block
// (see assets/shaders/texture_compress.slang). It submits to the context's background queue, so it never waits on or
// competes for a frame's queues in the CPU; use it from one thread at a time.
class GpuTextureEncoder {
public:
    struct Result {
        // Per input level: one 16-byte block per 4x4 texels, row major.
        std::vector<std::vector<std::byte>> blocks;
        // Time between the dispatch's start and end on the GPU.
        double gpu_ms = 0.0;
        // Wall time of the whole call: copy in, submit, wait, copy out.
        double total_ms = 0.0;
    };

    GpuTextureEncoder();
    ~GpuTextureEncoder();

    GpuTextureEncoder(GpuTextureEncoder const &) = delete;
    auto operator=(GpuTextureEncoder const &) -> GpuTextureEncoder & = delete;
    GpuTextureEncoder(GpuTextureEncoder &&other) noexcept;
    auto operator=(GpuTextureEncoder &&other) noexcept -> GpuTextureEncoder &;

    // Fails when the device has no shader objects or no spare queue, or the shader does not compile.
    [[nodiscard]] static auto create(VulkanContext &context, renderer::SlangCompiler const &compiler)
            -> std::expected<GpuTextureEncoder, std::string>;

    [[nodiscard]] auto encode_batch(std::span<GpuTextureLevel const> levels,
                                    GpuBlockFormat format = GpuBlockFormat::bc7) -> std::expected<Result, std::string>;

    [[nodiscard]] auto encode(std::span<std::byte const> rgba8, std::uint32_t width, std::uint32_t height,
                              GpuBlockFormat format = GpuBlockFormat::bc7) -> std::expected<Result, std::string>;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// For --gpu-texture-bench. With an image: encodes it on the GPU and with the engine's CPU path (UASTC, transcoded to
// BC7) and prints the times and how close each is to the original (PSNR over RGBA). With a directory: encodes every PNG
// and JPEG in it, once as a batch and once one at a time, and prints the totals.
[[nodiscard]] auto run_gpu_texture_benchmark(VulkanContext &context, renderer::SlangCompiler const &compiler,
                                             std::filesystem::path const &path) -> bool;
