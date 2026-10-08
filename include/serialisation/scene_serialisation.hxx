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

struct SceneCaptureReport {
    std::vector<std::string> warnings;
};

[[nodiscard]]
auto capture_scene(Scene const &scene, Renderer &renderer, EngineModels const &engine_models,
                   SceneCaptureReport *report = nullptr) -> SceneDescription;

struct SceneInstantiateOptions {
    std::vector<std::shared_ptr<AssetPack const>> packs;

    bool stream_models = false;
};

struct DeferredSlotOverride {
    entt::entity entity = entt::null;
    ModelHandle model{};
    std::vector<std::pair<std::uint32_t, MaterialHandle>> slots;
};

struct SceneInstantiateReport {
    std::uint32_t models_from_packs = 0;
    std::uint32_t models_from_source = 0;
    std::uint32_t models_reused = 0;
    std::uint32_t textures_from_packs = 0;
    std::uint32_t textures_from_source = 0;
    std::vector<std::string> warnings;

    std::vector<ModelHandle> streaming_models;
    std::vector<DeferredSlotOverride> deferred_slot_overrides;
};

auto apply_deferred_slot_overrides(Scene &scene, Renderer &renderer, std::vector<DeferredSlotOverride> &pending)
        -> bool;

[[nodiscard]]
auto instantiate_scene(Scene &scene, Renderer &renderer, EngineModels const &engine_models,
                       SceneDescription const &description, SceneInstantiateOptions const &options = {})
        -> std::expected<SceneInstantiateReport, LbfError>;

struct SceneSaveOptions {
    bool embed_assets = true;

    LbfWriteOptions write{};

    std::vector<std::shared_ptr<AssetPack const>> source_packs;
};

struct SceneSaveResult {
    std::uint64_t file_size = 0;
    std::uint64_t fingerprint = 0;
    AssetCookReport cook;
    std::vector<std::string> warnings;
    double seconds = 0.0;
};

[[nodiscard]]
auto save_scene(Scene const &scene, Renderer &renderer, EngineModels const &engine_models,
                std::filesystem::path const &path, SceneSaveOptions const &options = {})
        -> std::expected<SceneSaveResult, LbfError>;

struct SceneLoadResult {
    std::shared_ptr<AssetPack> pack;
    SceneInstantiateReport instantiate;
    SceneDecodeReport decode;
    double seconds = 0.0;
};

struct SceneLoadOptions {
    bool wait_for_textures = false;
};

[[nodiscard]]
auto load_scene(Scene &scene, Renderer &renderer, EngineModels const &engine_models, std::filesystem::path const &path,
                SceneLoadOptions const &options = {}) -> std::expected<SceneLoadResult, LbfError>;

class SceneSaveJob {
public:
    [[nodiscard]]
    static auto start(Scene const &scene, Renderer &renderer, EngineModels const &engine_models,
                      std::filesystem::path path, SceneSaveOptions options = {}) -> SceneSaveJob;

    [[nodiscard]] auto ready() const -> bool;

    [[nodiscard]] auto take() -> std::expected<SceneSaveResult, LbfError>;

    [[nodiscard]] auto path() const noexcept -> std::filesystem::path const & { return path_; }

    [[nodiscard]] auto fingerprint() const noexcept -> std::uint64_t { return fingerprint_; }

private:
    std::filesystem::path path_;
    std::uint64_t fingerprint_ = 0;
    std::future<std::expected<SceneSaveResult, LbfError>> future_;
};

class SceneLoadJob {
public:
    enum class Phase : std::uint8_t {
        reading,
        streaming,
        done,
    };

    [[nodiscard]]
    static auto start(std::filesystem::path path) -> SceneLoadJob;

    [[nodiscard]]
    auto step(Scene &scene, Renderer &renderer, EngineModels const &engine_models)
            -> std::optional<std::expected<SceneLoadResult, LbfError>>;

    [[nodiscard]] auto phase() const noexcept -> Phase { return phase_; }
    [[nodiscard]] auto path() const noexcept -> std::filesystem::path const & { return path_; }

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

[[nodiscard]]
auto scene_fingerprint(SceneDescription const &description) -> std::uint64_t;

inline constexpr std::string_view scene_file_extension = ".lbf";
