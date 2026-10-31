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

// One image, or one mip level, to compress: 8-bit RGBA, rows one after another.
struct Bc7Level {
    std::span<std::byte const> rgba;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

// Number of 16-byte BC7 blocks an image of this size compresses to.
[[nodiscard]] constexpr auto bc7_block_count(std::uint32_t width, std::uint32_t height) noexcept -> std::size_t {
    return static_cast<std::size_t>((width + 3) / 4) * ((height + 3) / 4);
}

// Compresses RGBA8 images to BC7 on the GPU, a whole batch in one dispatch: one compute thread per 4x4 block, mode 6
// only (see assets/shaders/bc7_encode.slang). It submits to the context's background queue, so it never waits on or
// competes for a frame's queues in the CPU; use it from one thread at a time.
class Bc7GpuEncoder {
public:
    struct Result {
        // Per input level: one 16-byte block per 4x4 texels, row major.
        std::vector<std::vector<std::byte>> blocks;
        // Time between the dispatch's start and end on the GPU.
        double gpu_ms = 0.0;
        // Wall time of the whole call: copy in, submit, wait, copy out.
        double total_ms = 0.0;
    };

    Bc7GpuEncoder();
    ~Bc7GpuEncoder();

    Bc7GpuEncoder(Bc7GpuEncoder const &) = delete;
    auto operator=(Bc7GpuEncoder const &) -> Bc7GpuEncoder & = delete;
    Bc7GpuEncoder(Bc7GpuEncoder &&other) noexcept;
    auto operator=(Bc7GpuEncoder &&other) noexcept -> Bc7GpuEncoder &;

    // Fails when the device has no shader objects or no spare queue, or the shader does not compile.
    [[nodiscard]] static auto create(VulkanContext &context, renderer::SlangCompiler const &compiler)
            -> std::expected<Bc7GpuEncoder, std::string>;

    [[nodiscard]] auto encode_batch(std::span<Bc7Level const> levels) -> std::expected<Result, std::string>;

    [[nodiscard]] auto encode(std::span<std::byte const> rgba8, std::uint32_t width, std::uint32_t height)
            -> std::expected<Result, std::string>;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// For --bc7-bench. With an image: encodes it on the GPU and with the engine's CPU path (UASTC, transcoded to BC7) and
// prints the times and how close each is to the original (PSNR over RGBA). With a directory: encodes every PNG and
// JPEG in it, once as a batch and once one at a time, and prints the totals.
[[nodiscard]] auto run_bc7_benchmark(VulkanContext &context, renderer::SlangCompiler const &compiler,
                                     std::filesystem::path const &path) -> bool;
