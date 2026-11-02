#pragma once

#include <cstdint>
#include <memory>

#include "assets/material.hxx"
#include "assets/model.hxx"

class ScriptStorage;

// What a Scene needs from whatever owns the GPU resources its entities refer to: the storage of its native scripts,
// reference counts on the materials and models they use, and a note that the lights changed. The Renderer implements it;
// HeadlessSceneResources is the same with nothing behind it, so a Scene (and the scripts and physics on it) can run with no
// renderer, no window and no GPU.
class ISceneResources {
public:
    virtual ~ISceneResources() = default;

    [[nodiscard]] virtual auto script_storage() noexcept -> ScriptStorage & = 0;

    virtual auto retain_material(MaterialHandle handle) -> void = 0;
    virtual auto release_material(MaterialHandle handle) -> void = 0;
    virtual auto release_model(ModelHandle handle) -> void = 0;

    virtual auto mark_lights_dirty() -> void = 0;
};

// Scene resources with no GPU: scripts are stored, nothing is reference counted and no lights need uploading.
class HeadlessSceneResources final : public ISceneResources {
public:
    HeadlessSceneResources();
    ~HeadlessSceneResources() override;

    HeadlessSceneResources(HeadlessSceneResources const &) = delete;
    auto operator=(HeadlessSceneResources const &) -> HeadlessSceneResources & = delete;

    [[nodiscard]] auto script_storage() noexcept -> ScriptStorage & override;

    auto retain_material(MaterialHandle) -> void override {}
    auto release_material(MaterialHandle) -> void override {}
    auto release_model(ModelHandle) -> void override {}
    auto mark_lights_dirty() -> void override {}

private:
    std::unique_ptr<ScriptStorage> scripts_;
};
