#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rendering/effect_manifest.hxx"
#include "rendering/game_gpu.hxx"
#include "rendering/game_graph.hxx"

using EffectShaderId = std::uint32_t;
using EffectId = std::uint32_t;
using EffectBufferId = std::uint32_t;

[[nodiscard]] constexpr auto game_slot_name(GameSlot slot) noexcept -> std::string_view {
    switch (slot) {
        case GameSlot::frame_start:
            return "frame_start";
        case GameSlot::after_depth:
            return "after_depth";
        case GameSlot::after_lighting:
            return "after_lighting";
        case GameSlot::before_composite:
            return "before_composite";
    }
    return "unknown";
}

[[nodiscard]] constexpr auto parse_game_slot(std::string_view name) noexcept -> std::optional<GameSlot> {
    for (auto const slot: {GameSlot::frame_start, GameSlot::after_depth, GameSlot::after_lighting,
                           GameSlot::before_composite}) {
        if (game_slot_name(slot) == name) {
            return slot;
        }
    }
    return std::nullopt;
}

// What a script gave for an effect's name: numbers for a param, a source name for an image input, or a buffer.
struct EffectValue {
    std::vector<double> numbers;
    std::optional<std::string> text;
    std::optional<EffectBufferId> buffer;
};

// Compute effects a script can attach to the frame without ever seeing a pass, handle or buffer. A script loads a
// shader's manifest (rendering/effect_manifest.hxx), makes instances of it with values for its params, inputs and
// buffers, and adds them to a GameSlot. Everything it can name is checked against the manifest, and the declaration
// the engine turns it into goes through GameGraph::add_compute, so the same enforcement applies: an effect whose
// declaration is rejected is dropped for the frame (and only it), with the reason kept in problem().
class EffectSystem {
public:
    // Registers a manifest's shader; tests supply their own.
    using Registrar = std::function<std::expected<GameComputeShader, std::string>(EffectManifest const &)>;

    // Registers shaders with `gpu` and releases the buffers of dropped effects through it.
    auto create(GameGpu &gpu) -> void;
    auto use_registrar(Registrar registrar) -> void { registrar_ = std::move(registrar); }

    // --- shaders

    // Reads and registers the manifest at `path` (relative to the data directory, under assets/, ending in .json). The
    // same path gives the same shader.
    [[nodiscard]] auto load(std::string_view path) -> std::expected<EffectShaderId, std::string>;
    // The same for a file by its absolute path; `name` is what messages call it, and edits to the file are noticed.
    [[nodiscard]] auto load_file(std::filesystem::path const &file, std::string name)
            -> std::expected<EffectShaderId, std::string>;
    // Registers a manifest from its text. `name` identifies it in messages.
    [[nodiscard]] auto define(std::string name, std::string_view manifest_json)
            -> std::expected<EffectShaderId, std::string>;
    // Reads the manifest again. A manifest that no longer parses leaves the shader as it was. Instances keep the
    // values of params and bindings that still exist with the same type. (The shader source itself is reloaded by
    // the engine's shader watcher.)
    [[nodiscard]] auto reload(EffectShaderId shader) -> std::expected<void, std::string>;
    [[nodiscard]] auto manifest(EffectShaderId shader) const noexcept -> EffectManifest const *;

    // --- instances

    [[nodiscard]] auto instance(EffectShaderId shader) -> std::expected<EffectId, std::string>;
    [[nodiscard]] auto set(EffectId effect, std::string_view name, EffectValue const &value)
            -> std::expected<void, std::string>;
    [[nodiscard]] auto add(EffectId effect, GameSlot slot) -> std::expected<void, std::string>;
    auto remove(EffectId effect) -> void;
    // Removes the effect and lets go of its buffers.
    auto destroy(EffectId effect) -> void;
    // Why the effect was last dropped from a frame, or empty.
    [[nodiscard]] auto problem(EffectId effect) const -> std::string;
    [[nodiscard]] auto exists(EffectId effect) const noexcept -> bool { return effects_.contains(effect); }
    [[nodiscard]] auto slot_of(EffectId effect) const noexcept -> std::optional<GameSlot>;

    // --- buffers a script can create: an opaque run of floats only effects can use

    [[nodiscard]] auto create_buffer(std::uint32_t elements) -> std::expected<EffectBufferId, std::string>;
    auto retain_buffer(EffectBufferId buffer) -> void;
    auto release_buffer(EffectBufferId buffer) -> void;
    [[nodiscard]] auto buffer_elements(EffectBufferId buffer) const noexcept -> std::uint32_t;

    // --- per frame

    // Declares the effects added to graph.slot(), in the order they were added. At frame_start it also checks the
    // manifests for edits (every so often) and frees the buffers nobody holds.
    auto declare(GameGraph &graph) -> void;

    // Reloads every shader whose manifest file changed on disk.
    auto poll_manifests() -> void;

private:
    struct Shader {
        std::string name;
        EffectManifest manifest;
        GameComputeShader compute{};
        std::filesystem::path file;
        std::filesystem::file_time_type modified{};
    };

    struct Effect {
        EffectShaderId shader = 0;
        // By manifest param and binding index.
        std::vector<std::array<double, 4>> params;
        std::vector<EffectSource> sources;
        std::vector<std::optional<EffectBufferId>> buffers;
        std::optional<GameSlot> slot;
        std::string problem;
    };

    struct Buffer {
        std::uint32_t elements = 0;
        std::uint32_t references = 0;
    };

    [[nodiscard]] static auto source_available(EffectSource source, GameSlot slot) noexcept -> bool;
    [[nodiscard]] auto validate(Effect const &effect, GameSlot slot) const -> std::optional<std::string>;
    [[nodiscard]] auto fit(Effect const &old_effect, EffectManifest const &old_manifest, EffectManifest const &manifest)
            -> Effect;
    [[nodiscard]] auto owned_buffer_name(EffectId id, std::string_view binding) const -> std::string;
    auto declare_effect(GameGraph &graph, EffectId id, Effect &effect) -> void;
    auto free_buffer_memory(std::string_view name) -> void;

    GameGpu *gpu_ = nullptr;
    Registrar registrar_;
    std::vector<Shader> shaders_;
    std::map<EffectId, Effect> effects_;
    std::vector<EffectId> order_;
    std::map<EffectBufferId, Buffer> buffers_;
    std::vector<std::string> unused_buffers_;
    EffectId next_effect_ = 1;
    EffectBufferId next_buffer_ = 1;
    std::uint32_t frames_ = 0;
};
