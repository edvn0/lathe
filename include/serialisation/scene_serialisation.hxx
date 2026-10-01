#pragma once

#include <chrono>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <entt/entt.hpp>

#include "core/forward.hxx"
#include "rendering/engine_models.hxx"
#include "serialisation/asset_pack.hxx"
#include "serialisation/lbf_container.hxx"
#include "serialisation/scene_codec.hxx"

class Scene;

// Saving and loading a live Scene as an .lbf file.
//
// Save: capture_scene() turns the registry into a SceneDescription (handles become asset references), then every
// referenced model and texture is cooked into MODL/TEXR chunks next to the SCEN chunk, so the file is self-contained.
//
// Load: the SCEN chunk is decoded, models are decoded from their MODL chunks on the thread pool and uploaded (no glTF
// parsing, tangents, LODs or meshlet building), textures stream from their TEXR chunks (no transcoding), and the
// entities are rebuilt. Assets missing from the file load from their source paths instead.

struct SceneCaptureReport {
    std::vector<std::string> warnings; // components that couldn't be expressed, e.g. procedural models
};

// Render thread (reads Renderer state).
[[nodiscard]]
auto capture_scene(Scene const &scene, Renderer &renderer, EngineModels const &engine_models,
                   SceneCaptureReport *report = nullptr) -> SceneDescription;

struct SceneInstantiateOptions {
    // Searched in order for cooked assets; usually the file the scene came from.
    std::vector<std::shared_ptr<AssetPack const>> packs;

    // Load models through Renderer::model_streamer() instead of uploading them before returning: entities appear at
    // once on the engine cube and swap to their models as uploads finish, a few primitives per frame. Per-slot
    // material overrides need the real model's materials, so they come back in
    // SceneInstantiateReport::deferred_slot_overrides for apply_deferred_slot_overrides().
    bool stream_models = false;
};

// A per-slot material override waiting for its model to finish streaming. Holds a reference on each material.
struct DeferredSlotOverride {
    entt::entity entity = entt::null;
    ModelHandle model{};
    std::vector<std::pair<std::uint32_t, MaterialHandle>> slots; // (source slot, material)
};

struct SceneInstantiateReport {
    std::uint32_t models_from_packs = 0;
    std::uint32_t models_from_source = 0;
    std::uint32_t models_reused = 0; // already loaded
    std::uint32_t textures_from_packs = 0;
    std::uint32_t textures_from_source = 0;
    std::vector<std::string> warnings;

    // stream_models only.
    std::vector<ModelHandle> streaming_models;
    std::vector<DeferredSlotOverride> deferred_slot_overrides;
};

// Applies the overrides whose models are no longer loading, dropping their material references, and removes them
// from `pending`. Returns true once none remain. Render thread.
auto apply_deferred_slot_overrides(Scene &scene, Renderer &renderer, std::vector<DeferredSlotOverride> &pending)
        -> bool;

// Replaces the scene's entities with `description`'s. Waits for the GPU to go idle first. Render thread.
[[nodiscard]]
auto instantiate_scene(Scene &scene, Renderer &renderer, EngineModels const &engine_models,
                       SceneDescription const &description, SceneInstantiateOptions const &options = {})
        -> std::expected<SceneInstantiateReport, LbfError>;

struct SceneSaveOptions {
    // Cook models and textures into the file. Without it the scene only references source paths (small and fast to
    // write, but loading then pays the full import cost).
    bool embed_assets = true;

    LbfWriteOptions write{};

    // Packs to copy already-cooked assets from instead of cooking them again (see AssetCookOptions::source_packs).
    std::vector<std::shared_ptr<AssetPack const>> source_packs;
};

struct SceneSaveResult {
    std::uint64_t file_size = 0;
    std::uint64_t fingerprint = 0; // scene_fingerprint() of what was saved
    AssetCookReport cook;
    std::vector<std::string> warnings;
    double seconds = 0.0;
};

// Render thread; cooks on thread_pool().
[[nodiscard]]
auto save_scene(Scene const &scene, Renderer &renderer, EngineModels const &engine_models,
                std::filesystem::path const &path, SceneSaveOptions const &options = {})
        -> std::expected<SceneSaveResult, LbfError>;

struct SceneLoadResult {
    // The opened file; pass it to SceneSaveOptions::source_packs when saving again so unchanged assets are copied.
    std::shared_ptr<AssetPack> pack;
    SceneInstantiateReport instantiate;
    SceneDecodeReport decode;
    double seconds = 0.0;
};

struct SceneLoadOptions {
    // Also block until every texture (cooked or from source) is uploaded, so nothing renders a fallback. Without it,
    // textures stream in over the next frames as usual.
    bool wait_for_textures = false;
};

// Blocking load: reads the file, decodes cooked models in parallel and uploads them before returning, so every
// entity has its real model on return. Render thread, outside frame recording. For a load that doesn't stall the
// frame, use SceneLoadJob.
[[nodiscard]]
auto load_scene(Scene &scene, Renderer &renderer, EngineModels const &engine_models, std::filesystem::path const &path,
                SceneLoadOptions const &options = {}) -> std::expected<SceneLoadResult, LbfError>;

// save_scene() without blocking: the capture happens in start(), on the calling (render) thread, and the cooking,
// compression and write run on a background thread. The Renderer must outlive the job; destroying an unfinished job
// waits for it.
class SceneSaveJob {
public:
    [[nodiscard]]
    static auto start(Scene const &scene, Renderer &renderer, EngineModels const &engine_models,
                      std::filesystem::path path, SceneSaveOptions options = {}) -> SceneSaveJob;

    [[nodiscard]] auto ready() const -> bool;

    // Call once ready().
    [[nodiscard]] auto take() -> std::expected<SceneSaveResult, LbfError>;

    [[nodiscard]] auto path() const noexcept -> std::filesystem::path const & { return path_; }

    // Fingerprint of the captured scene, known from start(): what the scene will match once the save succeeds.
    [[nodiscard]] auto fingerprint() const noexcept -> std::uint64_t { return fingerprint_; }

private:
    std::filesystem::path path_;
    std::uint64_t fingerprint_ = 0;
    std::future<std::expected<SceneSaveResult, LbfError>> future_;
};

// load_scene() without blocking: the file is opened and its SCEN chunk decoded on a background thread, then step()
// instantiates the entities with streamed models and finishes once every model has installed.
class SceneLoadJob {
public:
    enum class Phase : std::uint8_t {
        reading,
        streaming, // entities exist; models are still uploading
        done,
    };

    [[nodiscard]]
    static auto start(std::filesystem::path path) -> SceneLoadJob;

    // Render thread, once per frame. Returns the result on the frame the load finishes or fails.
    [[nodiscard]]
    auto step(Scene &scene, Renderer &renderer, EngineModels const &engine_models)
            -> std::optional<std::expected<SceneLoadResult, LbfError>>;

    [[nodiscard]] auto phase() const noexcept -> Phase { return phase_; }
    [[nodiscard]] auto path() const noexcept -> std::filesystem::path const & { return path_; }

    // Models still uploading, for progress display.
    [[nodiscard]] auto models_remaining(Renderer &renderer) const -> std::size_t;
    [[nodiscard]] auto models_total() const noexcept -> std::size_t { return streaming_.size(); }

private:
    struct Prepared {
        std::shared_ptr<AssetPack> pack;
        SceneDescription description;
        SceneDecodeReport decode;
    };

    std::filesystem::path path_;
    Phase phase_ = Phase::reading;
    std::chrono::steady_clock::time_point started_at_{};
    std::future<std::expected<Prepared, LbfError>> future_;
    std::optional<SceneLoadResult> result_;
    std::vector<ModelHandle> streaming_;
    std::vector<DeferredSlotOverride> deferred_;
};

// Identity of a scene's content that ignores entity order, so the editor can tell whether the scene changed since
// it was loaded or saved.
[[nodiscard]]
auto scene_fingerprint(SceneDescription const &description) -> std::uint64_t;

inline constexpr std::string_view scene_file_extension = ".lbf";
