#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <unordered_map>
#include <vector>

#include <entt/entt.hpp>

#include "core/renderer_error.hxx"
#include "rendering/game_gpu.hxx"
#include "rendering/game_graph.hxx"
#include "scene/components.hxx"

struct MaterialStorage;

// Simulates and draws every entity's Components::ParticleEmitter. It is built on the same GameGraph a game uses: per
// emitter a compute pass advances the particles in a persistent buffer, and a scene draw renders them inside the
// forward pass. The engine owns the buffers; Lua and games only ever set the component's data.
class ParticleSystem {
public:
    // Bytes of one particle in the shaders (two float4).
    static constexpr VkDeviceSize particle_bytes = 32;
    // Frame time beyond this is not simulated, so a hitch does not fling the whole buffer at once.
    static constexpr float max_delta_time = 0.1F;

    // Registers the shaders. Safe to call again.
    auto create(GameGpu &gpu) -> std::expected<void, RendererError>;

    // Uses these shaders without a GameGpu (tests); buffers of removed emitters are then not released.
    auto use_shaders(GameComputeShader simulate, GameGraphicsShader draw) noexcept -> void;

    // Declares this frame's passes for the emitters in `registry`. Does nothing outside GameSlot::frame_start or before
    // the shaders exist. `materials` tints particles by their emitter's material, and may be null.
    auto declare(GameGraph &graph, entt::registry const &registry, MaterialStorage const *materials, float delta_time)
            -> void;

    [[nodiscard]] auto emitter_count() const noexcept -> std::size_t { return states_.size(); }

    // The emitter with its numbers brought into the range the shaders and buffers support.
    [[nodiscard]] static auto sanitised(Components::ParticleEmitter emitter) -> Components::ParticleEmitter;

private:
    struct State {
        std::string buffer;
        std::uint32_t count = 0;
        // The ring slot the next spawned particle takes.
        std::uint32_t head = 0;
        std::uint32_t frame = 0;
        float carry = 0.0F;
        bool seen = false;
    };

    GameGpu *gpu_ = nullptr;
    GameComputeShader simulate_{};
    GameGraphicsShader draw_{};
    bool ready_ = false;
    std::unordered_map<entt::entity, State> states_;
};
