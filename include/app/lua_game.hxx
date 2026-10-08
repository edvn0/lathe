#pragma once

#include <memory>
#include <string>

#include <glm/mat4x4.hpp>

#include "app/game.hxx"
#include "scripting/lua_api.hxx"

class LuaRuntime;

struct LuaGameHost {
    Scene *scene = nullptr;
    Renderer *renderer = nullptr;
    EngineModels const *engine_models = nullptr;

    GameHost host;

    glm::mat4 inverse_view_projection{1.0F};
};

class LuaGame final : public IGame {
public:
    LuaGame();
    ~LuaGame() override;

    auto add_native_module(std::string name, lua_CFunction opener) -> void;

    auto attach_host(GameHost host) -> void override;

    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;

    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void override;
    auto on_mouse_button_pressed(Scene &scene, MouseButtonPressedEvent const &event) -> void override;

    [[nodiscard]] auto wants_cursor() const -> bool override;
    auto on_cursor_position(Scene &scene, CursorPositionEvent const &event) -> void override;

    auto on_ui(Scene &scene, Renderer &renderer) -> void override;

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};
