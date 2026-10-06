#pragma once

#include <memory>
#include <string>

#include <glm/mat4x4.hpp>

#include "app/game.hxx"
#include "scripting/lua_api.hxx"

class LuaRuntime;

// What the Lua bindings reach through LuaRuntime::host(): the scene and renderer of the callback in flight, the host
// services, and the camera the game last asked for (for picking).
struct LuaGameHost {
    Scene *scene = nullptr;
    Renderer *renderer = nullptr;
    EngineModels const *engine_models = nullptr;

    GameHost host;

    glm::mat4 inverse_view_projection{1.0F};
};

// A game written in Lua. Its entry script (GameHost::script_entry, default assets/scripts/main.lua under the data root; require("a.b") loads assets/scripts/a/b.lua)
// returns a table of callbacks:
//
//   on_populate()             build the scene (scene.spawn, assets.load_model, ...); also runs on Ctrl+R, which
//                             reloads every script.
//   on_bind()                 the active scene changed (editor scene -> runtime scene); look entities up again.
//   on_update(dt)             every frame while playing.
//   on_key(key, modifiers)    a key went down.
//   on_mouse_button(button)   a mouse button went down.
//   on_cursor(ndc_x, ndc_y, inside)   when `wants_cursor` is true.
//   on_ui()                   ImGui, through the ui table.
//   camera(aspect)            returns { eye = {x,y,z}, target = {x,y,z}, fov = degrees, near = n, far = f }.
//   wants_cursor              a boolean field: keep the cursor free and visible.
//
// An error in a callback is logged and shown on screen, and that callback stops running until the next reload.
class LuaGame final : public IGame {
public:
    LuaGame();
    ~LuaGame() override;

    // require(name) in the game's scripts returns what `opener` leaves on the stack (see luaL_requiref). Add before
    // on_populate().
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
