#pragma once

#include <volk.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "core/forward.hxx"

namespace renderer {
    class SlangCompiler;
}

// Prototype: compresses an RGBA8 image to BC7 on the GPU, one compute thread per 4x4 block (mode 6 only; see
// assets/shaders/bc7_encode.slang). Runs on the graphics queue through VulkanContext::one_time_submit, so it is for
// tools and measurement for now.
class Bc7GpuEncoder {
public:
    struct Result {
        // One 16-byte block per 4x4 texels, row major.
        std::vector<std::byte> blocks;
        // Time between the dispatch's start and end on the GPU.
        double gpu_ms = 0.0;
        // Wall time of the whole call: copy in, submit, wait, copy out.
        double total_ms = 0.0;
    };

    Bc7GpuEncoder() = default;
    ~Bc7GpuEncoder();

    Bc7GpuEncoder(Bc7GpuEncoder const &) = delete;
    auto operator=(Bc7GpuEncoder const &) -> Bc7GpuEncoder & = delete;
    Bc7GpuEncoder(Bc7GpuEncoder &&other) noexcept;
    auto operator=(Bc7GpuEncoder &&other) noexcept -> Bc7GpuEncoder &;

    [[nodiscard]] static auto create(VulkanContext &context, renderer::SlangCompiler const &compiler)
            -> std::expected<Bc7GpuEncoder, std::string>;

    [[nodiscard]] auto encode(std::span<std::byte const> rgba8, std::uint32_t width, std::uint32_t height)
            -> std::expected<Result, std::string>;

private:
    auto destroy() noexcept -> void;

    VulkanContext *context_ = nullptr;
    VkShaderEXT shader_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkQueryPool timestamps_ = VK_NULL_HANDLE;
};

// Encodes 'image' on the GPU and with the engine's CPU path (UASTC, transcoded to BC7), then prints the times and how
// close each is to the original (PSNR over RGBA). For --bc7-bench.
[[nodiscard]] auto run_bc7_benchmark(VulkanContext &context, renderer::SlangCompiler const &compiler,
                                     std::filesystem::path const &image) -> bool;
