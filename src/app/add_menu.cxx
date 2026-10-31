// The editor's Add menu: Shift+A over a viewport opens a popup at the cursor that adds an entity where the cursor
// points. The entries are commands in Application::commands, so a later palette or key binding can run the same ones.

#include "app/application.hxx"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <imgui.h>

#include "app/asset_picker.hxx"
#include "rendering/entity.hxx"
#include "rendering/renderer.hxx"
#include "scene/components.hxx"
#include "scene/selection_context.hxx"

namespace {
    constexpr auto add_menu_popup_id = "viewport_add_menu";

    // The popup is a quick-pick at the cursor, so it is drawn larger and looser than the editor's panels.
    constexpr auto add_menu_font_scale = 1.4F;
    constexpr auto add_menu_min_width = 220.0F;

    // Loaded models listed at most this many at a time while filtering.
    constexpr auto add_menu_filtered_model_limit = std::size_t{12};

    struct AddMenuStyle {
        AddMenuStyle() {
            ImGui::PushFont(nullptr, ImGui::GetFontSize() * add_menu_font_scale);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{10.0F, 10.0F});
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{8.0F, 8.0F});
            ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 6.0F);
        }

        ~AddMenuStyle() {
            ImGui::PopStyleVar(3);
            ImGui::PopFont();
        }

        AddMenuStyle(AddMenuStyle const &) = delete;
        auto operator=(AddMenuStyle const &) -> AddMenuStyle & = delete;
    };

    // How far along the cursor ray an entity is put when the ray misses the ground plane.
    constexpr auto max_ground_distance = 500.0F;

    // Creates an entity at the position the menu was opened for, lets `configure` give it its components, and
    // selects it.
    template<typename Configure>
    auto spawn_at_menu_position(Application &app, std::string name, Configure &&configure) -> void {
        auto entity = Entity{app.active_scene(), std::move(name)};
        entity.emplace<Components::Transform>(Components::Transform{.position = app.add_menu_position});
        configure(entity);
        selection_context().select(entity);
    }
}

auto Application::register_commands() -> void {
    commands.add({.id = "add.empty", .label = "Empty", .category = "", .action = [this] {
                      spawn_at_menu_position(*this, "Entity", [](Entity &) {});
                  }});

    auto const add_primitive = [this](std::string id, std::string label, std::string name, ModelHandle EngineModels::*model) {
        commands.add({.id = std::move(id), .label = std::move(label), .category = "Mesh",
                      .action = [this, name = std::move(name), model] {
                          spawn_at_menu_position(*this, name, [&](Entity &entity) {
                              entity.emplace<Components::Model>(Components::Model{.model = engine_models.*model});
                          });
                      }});
    };
    add_primitive("add.cube", "Cube", "Cube", &EngineModels::cube);
    add_primitive("add.sphere", "Sphere", "Sphere", &EngineModels::sphere);
    add_primitive("add.capsule", "Capsule", "Capsule", &EngineModels::capsule);

    commands.add({.id = "add.point_light", .label = "Point Light", .category = "Light", .action = [this] {
                      spawn_at_menu_position(*this, "Point Light", [](Entity &entity) {
                          entity.emplace<Components::PointLight>();
                      });
                  }});
    commands.add({.id = "add.spot_light", .label = "Spot Light", .category = "Light", .action = [this] {
                      spawn_at_menu_position(*this, "Spot Light", [](Entity &entity) {
                          entity.emplace<Components::SpotLight>();
                      });
                  }});

    commands.add({.id = "add.model_file", .label = "Model from File...", .category = "Model",
                  .action = [this] { request_import_model(); }});
}

auto Application::cursor_spawn_position(ViewportPane const &pane) const -> glm::vec3 {
    auto const &camera = pane.camera;
    auto const aspect = pane.content_size.x / pane.content_size.y;
    auto const mouse = ImGui::GetIO().MousePos;
    auto const ndc_x = (((mouse.x - pane.screen_pos.x) / pane.content_size.x) * 2.0F) - 1.0F;
    auto const ndc_y = 1.0F - (((mouse.y - pane.screen_pos.y) / pane.content_size.y) * 2.0F);

    // Camera-space ray through the pixel (LH, +z forward), as the click picking builds it.
    auto const projection = camera.projection(aspect);
    auto const camera_ray = glm::vec3{ndc_x / projection[0][0], ndc_y / projection[1][1], 1.0F};
    auto const to_world = glm::inverse(camera.view());
    auto const direction = glm::normalize(glm::vec3{to_world * glm::vec4{camera_ray, 0.0F}});
    auto const origin = camera.position();

    // The ground plane (y = 0) under the cursor, if the ray reaches it.
    if (std::abs(direction.y) > 1e-4F) {
        auto const distance = -origin.y / direction.y;

        if (distance > 0.0F && distance < max_ground_distance) {
            return origin + (direction * distance);
        }
    }

    return origin + (direction * camera_pivot_distance);
}

auto Application::draw_add_menu() -> void {
    if (!ImGui::IsPopupOpen(add_menu_popup_id)) {
        return;
    }

    AddMenuStyle const style;

    ImGui::SetNextWindowSizeConstraints(ImVec2{add_menu_min_width, 0.0F}, ImVec2{FLT_MAX, FLT_MAX});
    if (!ImGui::BeginPopup(add_menu_popup_id)) {
        return;
    }

    if (ImGui::IsWindowAppearing()) {
        add_menu_filter.Clear();
        ImGui::SetKeyboardFocusHere();
    }
    add_menu_filter.Draw("##add_filter", 0.0F);
    ImGui::Separator();

    // Only the add commands are listed here; the others (file, gizmo, ...) have their own menus.
    auto all = std::vector<editor::Command>{};
    for (auto const &command: commands.commands()) {
        if (command.id.starts_with("add.")) {
            all.push_back(command);
        }
    }
    auto const &models = renderer->assets().models().entries();

    auto const spawn_loaded_model = [&](ModelHandle model, std::string name) {
        renderer->retain_model(model);
        spawn_at_menu_position(*this, std::move(name), [&](Entity &entity) {
            entity.emplace<Components::Model>(Components::Model{.model = model});
            entity.emplace<Components::StreamedModelTag>();
        });
        ImGui::CloseCurrentPopup();
    };

    // Typing lists every match in one flat list, the first highlighted and run by Enter; with nothing typed the
    // commands are grouped into submenus.
    if (add_menu_filter.IsActive()) {
        auto first = true;
        auto const enter = ImGui::IsKeyPressed(ImGuiKey_Enter, false);
        // True when the entry was clicked, or is the first match and Enter was pressed.
        auto const activated = [&](char const *label) {
            auto const clicked = ImGui::MenuItem(label, nullptr, first);
            auto const entered = first && enter;
            first = false;
            return clicked || entered;
        };

        for (auto const &command: all) {
            auto const label = command.category.empty() ? command.label : command.category + " > " + command.label;

            if (add_menu_filter.PassFilter(label.c_str()) && activated(label.c_str())) {
                commands.run(command.id);
                ImGui::CloseCurrentPopup();
            }
        }

        auto shown = std::size_t{0};
        for (auto index = std::size_t{0}; index < models.size() && shown < add_menu_filtered_model_limit; ++index) {
            auto const &entry = models[index];

            if (entry.name.empty() || !entry.handle.valid() || !add_menu_filter.PassFilter(entry.name.c_str())) {
                continue;
            }

            ImGui::PushID(static_cast<int>(index));
            auto const picked = activated(("Model > " + entry.name).c_str());
            ImGui::PopID();
            ++shown;

            if (picked) {
                spawn_loaded_model(entry.handle, entry.name);
            }
        }

        if (first) {
            ImGui::TextDisabled("Nothing matches");
        }

        ImGui::EndPopup();
        return;
    }

    auto const run_item = [&](editor::Command const &command) {
        if (ImGui::MenuItem(command.label.c_str(), editor::shortcut_text(command.shortcut).c_str(), false,
                            commands.is_enabled(command.id))) {
            commands.run(command.id);
        }
    };

    for (auto const &command: all) {
        if (command.category.empty()) {
            run_item(command);
        }
    }

    auto categories = std::vector<std::string_view>{};
    for (auto const &command: all) {
        if (!command.category.empty() && !std::ranges::contains(categories, std::string_view{command.category})) {
            categories.emplace_back(command.category);
        }
    }

    for (auto const category: categories) {
        if (!ImGui::BeginMenu(std::string{category}.c_str())) {
            continue;
        }

        for (auto const &command: all) {
            if (command.category == category) {
                run_item(command);
            }
        }

        // Models already loaded into the renderer, listed under Model with the file loader.
        if (category == "Model" && !models.empty()) {
            ImGui::Separator();
            if (ImGui::BeginMenu("Loaded")) {
                if (auto const picked = ui::draw_asset_choices(models, std::span<ModelHandle const>{}, ModelHandle{})) {
                    auto const named = std::ranges::find(models, *picked, &std::remove_cvref_t<decltype(models.front())>::handle);
                    spawn_loaded_model(*picked, named == models.end() ? std::string{} : named->name);
                }
                ImGui::EndMenu();
            }
        }

        ImGui::EndMenu();
    }

    ImGui::EndPopup();
}
