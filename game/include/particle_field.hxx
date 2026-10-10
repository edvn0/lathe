#pragma once

#include <cstdint>
#include <expected>

#include "core/renderer_error.hxx"
#include "rendering/game_gpu.hxx"
#include "rendering/game_graph.hxx"

// A GPU particle fountain, and the worked example of a game adding to the frame graph. A compute pass simulates the
// particles in a buffer the game owns, and a scene draw renders them inside the forward pass. Both are declared
// through the GameGraph, so neither can reach an engine resource except through what the graph hands out.
class ParticleField {
public:
    static constexpr std::uint32_t particle_count = 8192;

    // Registers the shaders and creates the buffer. Safe to call again; a field that is already set up is kept.
    auto create(GameGpu &gpu) -> std::expected<void, RendererError>;

    // Declares this frame's passes. Does nothing outside GameSlot::frame_start or before create() succeeded.
    auto declare(GameGraph &graph, float delta_time) -> void;

private:
    GameComputeShader simulate_{};
    GameGraphicsShader draw_{};
    GameBufferHandle particles_{};
    std::uint32_t frame_ = 0;
};
