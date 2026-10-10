#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "core/renderer_error.hxx"
#include "gpu/buffer.hxx"
#include "rendering/game_graph.hxx"

struct Renderer;

struct GameComputeShaderInfo {
    // Relative to the data directory, e.g. "assets/shaders/game/particles_simulate.slang".
    std::string source;
    std::string entry_point = "main_cs";
    std::string debug_name;
};

struct GameGraphicsShaderInfo {
    std::string source;
    std::string vertex_entry_point = "main_vs";
    std::string fragment_entry_point = "main_fs";
    std::string debug_name;
};

struct GameBufferInfo {
    VkDeviceSize size = 0;
    std::string debug_name;
};

// What a game may ask of the GPU: shaders that run through the game graph and buffers that survive between frames.
// It replaces reaching for Renderer::context() and the bindless table, which hand out the whole device.
class GameGpu {
public:
    explicit GameGpu(Renderer &renderer) noexcept : renderer_{&renderer} {}

    [[nodiscard]] auto register_compute(GameComputeShaderInfo const &info)
            -> std::expected<GameComputeShader, RendererError>;
    [[nodiscard]] auto register_graphics(GameGraphicsShaderInfo const &info)
            -> std::expected<GameGraphicsShader, RendererError>;

    // One device-local storage buffer with an address, in a copy for every frame in flight. Its contents start
    // zero-filled.
    [[nodiscard]] auto create_buffer(GameBufferInfo const &info) -> std::expected<GameBufferHandle, RendererError>;

    [[nodiscard]] auto buffer(GameBufferHandle handle, std::uint32_t frame_slot) const noexcept -> Buffer const *;
    [[nodiscard]] auto buffer_name(GameBufferHandle handle) const noexcept -> std::string_view;
    [[nodiscard]] auto memory() noexcept -> GameGraphMemory & { return memory_; }

private:
    struct Entry {
        std::vector<Buffer> copies;
        std::string name;
    };

    Renderer *renderer_;
    std::vector<Entry> buffers_;
    GameGraphMemory memory_;
};
