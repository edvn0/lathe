#include "app/application.hxx"

#include <csignal>
#include <memory>
#include <volk.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/vec2.hpp>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <entt/entt.hpp>

#include "core/allocator.hxx"
#include "core/config.hxx"
#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "glm/gtc/type_ptr.hpp"
#include "gpu/context.hxx"
#include "implot.h"
// DockBuilder* API.
#include "imgui_internal.h"
#include "rendering/debug_renderer.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/entity.hxx"
#include "rendering/imgui_renderer.hxx"
#include "rendering/imgui_widget.hxx"
#include "scene/components.hxx"
#include "scene/editor_camera.hxx"
#include "scene/selection_context.hxx"
#if MINGW_VULKAN_TRACK_MEMORY
#include "core/memory_tracking_ui.hxx"
#endif
#include "assets/shader_hot_reload_watcher.hxx"
#include "gpu/renderdoc.hxx"
#include "gpu/swapchain.hxx"
#include "physics/physics.hxx"
#include "physics/physics_world.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"

namespace {

    // GeneratedMeta wins over Meta if an entity somehow carries both.
    [[nodiscard]] auto entity_display_name(entt::registry const &registry, entt::entity entity) -> std::string {
        char const *raw = nullptr;
        if (auto const *generated = registry.try_get<Components::GeneratedMeta>(entity)) {
            raw = generated->name.c_str();
        } else if (auto const *meta = registry.try_get<Components::Meta>(entity)) {
            raw = meta->name.c_str();
        }
        if (raw != nullptr && raw[0] != '\0') {
            return raw;
        }
        return std::format("Entity {}", entt::to_integral(entity));
    }

    constexpr auto draw_point_light = [](Components::PointLight &point_light) -> bool {
        bool changed = false;
        changed |= ImGui::ColorEdit3("Colour", &point_light.colour.x);
        changed |= ImGui::SliderFloat("Intensity", &point_light.intensity, 0.0F, 200.0F);
        changed |= ImGui::SliderFloat("Range", &point_light.range, 0.5F, 100.0F);
        return changed;
    };

    constexpr auto draw_spot_light = [](Components::SpotLight &spot_light) -> bool {
        bool changed = false;
        changed |= ImGui::ColorEdit3("Colour", &spot_light.colour.x);
        changed |= ImGui::SliderFloat("Intensity", &spot_light.intensity, 0.0F, 200.0F);
        changed |= ImGui::SliderFloat("Range", &spot_light.range, 0.5F, 100.0F);
        changed |= ImGui::SliderFloat("Inner cone", &spot_light.inner_cone_degrees, 0.0F, 89.0F, "%.1f deg");
        changed |= ImGui::SliderFloat("Outer cone", &spot_light.outer_cone_degrees, 0.0F, 89.0F, "%.1f deg");
        return changed;
    };

    // Rotation is edited as Euler degrees and converted back to a quaternion on each change.
    constexpr auto draw_transform = [](Components::Transform &transform) -> bool {
        bool changed = false;
        changed |= ImGui::DragFloat3("Position", &transform.position.x, 0.1F);

        auto euler_degrees = glm::degrees(glm::eulerAngles(transform.rotation));
        if (ImGui::DragFloat3("Rotation", &euler_degrees.x, 0.5F, 0.0F, 0.0F, "%.1f deg")) {
            transform.rotation = glm::quat(glm::radians(euler_degrees));
            changed = true;
        }

        changed |= ImGui::DragFloat3("Scale", &transform.scale.x, 0.01F, 0.001F, 1000.0F);
        return changed;
    };

    constexpr auto draw_lifetime = [](Components::Lifetime &lifetime) -> bool {
        return ImGui::DragFloat("Remaining seconds", &lifetime.remaining_seconds, 0.05F, 0.0F, 3600.0F);
    };

    constexpr auto draw_rigid_body = [](Components::RigidBody &body) -> bool {
        bool changed = false;

        // Heightfield/compound shapes are generated from terrain/mesh data and can't be switched here.
        bool const generated_shape =
                body.shape == Components::BodyShape::heightfield || body.shape == Components::BodyShape::compound;

        if (generated_shape) {
            ImGui::TextDisabled("Shape: %s (generated, not editable)",
                                body.shape == Components::BodyShape::heightfield ? "Heightfield" : "Compound");
        } else {
            int shape_index = body.shape == Components::BodyShape::capsule ? 1 : 0;
            constexpr std::array<char const *, 2> shape_names{"Box", "Capsule"};
            if (ImGui::Combo("Shape", &shape_index, shape_names.data(), static_cast<int>(shape_names.size()))) {
                body.shape = shape_index == 1 ? Components::BodyShape::capsule : Components::BodyShape::box;
                changed = true;
            }

            if (body.shape == Components::BodyShape::box) {
                changed |= ImGui::DragFloat3("Half extents", &body.half_extents.x, 0.05F, 0.01F, 1000.0F);
            } else {
                changed |= ImGui::DragFloat("Capsule radius", &body.capsule_radius, 0.05F, 0.01F, 1000.0F);
                changed |= ImGui::DragFloat("Capsule height", &body.capsule_height, 0.05F, 0.01F, 1000.0F);
            }
        }

        changed |= ImGui::Checkbox("Static", &body.is_static);
        ImGui::SameLine();
        changed |= ImGui::Checkbox("Lock rotation", &body.lock_rotation);

        ImGui::BeginDisabled(body.is_static);
        changed |= ImGui::DragFloat("Mass", &body.mass, 0.1F, 0.01F, 10000.0F);
        ImGui::EndDisabled();

        changed |= ImGui::SliderFloat("Restitution", &body.restitution, 0.0F, 1.0F);
        changed |= ImGui::DragFloat3("Initial velocity", &body.velocity.x, 0.1F);

        return changed;
    };

    constexpr auto draw_rows = [](auto &index, entt::registry &registry, auto &&view, auto &&draw_light) {
        for (auto [entity, transform, light, meta]: view.each()) {
            ImGui::PushID(static_cast<int>(index++));
            if (ImGui::TreeNode(meta.name.c_str())) {
                bool changed = ImGui::DragFloat3("Position", &transform.position.x, 0.1F);
                changed |= draw_light(light);

                if (changed) {
                    using LightT = std::decay_t<decltype(light)>;
                    registry.patch<LightT>(entity);
                }

                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    };

    // Copies every component in Cs... that `source` has onto `dest`. Never pass Components::PhysicsBody: it's a
    // non-owning handle into PhysicsWorld, so a copy would share the original's rigid body.
    template<typename... Cs>
    auto copy_components(entt::registry &registry, entt::entity source, entt::entity dest) -> void {
        (
                [&] {
                    // Empty tag types have no storage, so try_get<T>() doesn't compile for them.
                    if constexpr (std::is_empty_v<Cs>) {
                        if (registry.all_of<Cs>(source)) {
                            registry.emplace<Cs>(dest);
                        }
                    } else if (auto const *component = registry.try_get<Cs>(source)) {
                        registry.emplace<Cs>(dest, *component);
                    }
                }(),
                ...);
    }

    template<std::size_t N>
    auto copy_to_buffer(std::array<char, N> &buffer, std::string_view text) -> void {
        auto const length = std::min(text.size(), N - 1);
        std::copy_n(text.begin(), length, buffer.begin());
        buffer[length] = '\0';
    }

    // Writes to GeneratedMeta when the entity has one, since it wins in entity_display_name(), else to Meta.
    // Surrounding whitespace is trimmed; an empty name is ignored.
    auto rename_entity(entt::registry &registry, entt::entity entity, std::string_view name) -> void {
        auto const is_space = [](unsigned char c) { return std::isspace(c) != 0; };
        while (!name.empty() && is_space(static_cast<unsigned char>(name.front()))) {
            name.remove_prefix(1);
        }
        while (!name.empty() && is_space(static_cast<unsigned char>(name.back()))) {
            name.remove_suffix(1);
        }

        if (name.empty() || !registry.valid(entity)) {
            return;
        }

        if (registry.all_of<Components::GeneratedMeta>(entity)) {
            registry.patch<Components::GeneratedMeta>(
                    entity, [&](Components::GeneratedMeta &meta) { meta.name = std::string{name}; });
        } else {
            registry.emplace_or_replace<Components::Meta>(entity, Components::Meta{.name = FlyString{name}});
        }
    }

    // `entities` without the ones that have a selected ancestor, order kept. Moving, duplicating or deleting an
    // entity carries its subtree, so acting on those descendants as well would repeat the work.
    [[nodiscard]] auto selection_roots(entt::registry const &registry, std::span<entt::entity const> entities)
            -> std::vector<entt::entity> {
        // Bounds the walk, so a cyclic Parent chain terminates.
        constexpr std::size_t max_parent_depth = 1024;

        std::vector<entt::entity> roots;
        roots.reserve(entities.size());

        for (auto const entity: entities) {
            if (!registry.valid(entity)) {
                continue;
            }

            bool has_selected_ancestor = false;
            std::size_t depth = 0;
            for (auto const *parent = registry.try_get<Components::Parent>(entity);
                 parent != nullptr && registry.valid(parent->entity) && depth++ < max_parent_depth;
                 parent = registry.try_get<Components::Parent>(parent->entity)) {
                if (std::ranges::find(entities, parent->entity) != entities.end()) {
                    has_selected_ancestor = true;
                    break;
                }
            }

            if (!has_selected_ancestor) {
                roots.push_back(entity);
            }
        }

        return roots;
    }

    // Splits an affine matrix into position, rotation and scale; shear is dropped.
    auto set_transform_from_matrix(entt::registry &registry, entt::entity entity, glm::mat4 const &matrix) -> void {
        auto const translation = glm::vec3{matrix[3]};
        glm::vec3 const scale{glm::length(glm::vec3{matrix[0]}), glm::length(glm::vec3{matrix[1]}),
                              glm::length(glm::vec3{matrix[2]})};
        glm::mat3 const rotation_matrix{glm::vec3{matrix[0]} / scale.x, glm::vec3{matrix[1]} / scale.y,
                                        glm::vec3{matrix[2]} / scale.z};
        auto const rotation = glm::quat_cast(rotation_matrix);

        // patch<>() so Scene::on_transform_changed fires.
        registry.patch<Components::Transform>(entity, [&](Components::Transform &transform) {
            transform.position = translation;
            transform.rotation = rotation;
            transform.scale = scale;
        });
    }

    // Filters for the model file browser.
    [[nodiscard]] auto model_file_filters() -> std::vector<gui::FileBrowser::Filter> {
        return {
                {.label = "glTF models (*.gltf, *.glb)", .extensions = {".gltf", ".glb"}},
                {.label = "All files", .extensions = {}},
        };
    }

    using gui::widget;

    // Materials queued in pending_deletions can't be newly assigned while they wait to be destroyed.
    [[nodiscard]] auto material_deletion_label(std::string_view name) -> std::string {
        return std::format("material:{}", name);
    }

    [[nodiscard]] auto is_material_pending_deletion(std::span<Application::PendingDeletion const> pending_deletions,
                                                    std::string_view name) -> bool {
        auto const label = material_deletion_label(name);
        return std::ranges::any_of(pending_deletions,
                                   [&](Application::PendingDeletion const &pending) { return pending.label == label; });
    }
} // namespace

Application::Application(VulkanContext &ctx) noexcept :
    context(ctx), renderer(std::make_unique<Renderer>(context)),
    debug_renderer(std::make_unique<debug_draw::DebugRenderer>(*renderer)) {
    timing_buffers.fill(ScrollingBuffer{600});
}

Application::~Application() {
    if (terrain) {
        terrain->wait_all();
    }

    shader_watcher_.stop();
}

auto Application::on_ui(std::uint32_t frame_index) -> void {
    // Must match the CompositeTarget main.cxx passes to Renderer::record_frame.
    if (is_playing && play_fullscreen) {
        return;
    }

    auto const *main_viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(main_viewport->WorkPos);
    ImGui::SetNextWindowSize(main_viewport->WorkSize);
    ImGui::SetNextWindowViewport(main_viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("##dockspace_host", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                         ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBackground);
    ImGui::PopStyleVar(3);

    ImGuiID const dockspace_id = ImGui::GetID("MainDockSpace");

    // Build the default layout only when imgui.ini didn't restore one.
    if (ImGui::DockBuilderGetNode(dockspace_id) == nullptr) {
        ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dockspace_id, main_viewport->WorkSize);

        ImGuiID center = dockspace_id;
        ImGuiID const left = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.20F, nullptr, &center);
        ImGuiID const right = ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.25F, nullptr, &center);
        ImGuiID const bottom = ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, 0.28F, nullptr, &center);

        ImGui::DockBuilderDockWindow("Viewport", center);
        ImGui::DockBuilderDockWindow("Hierarchy", left);
        ImGui::DockBuilderDockWindow("Inspector", right);
        ImGui::DockBuilderDockWindow("Console", bottom);
        ImGui::DockBuilderDockWindow("Assets", bottom);
        ImGui::DockBuilderDockWindow("Load Model", bottom);
        ImGui::DockBuilderDockWindow("Simulation", bottom);
        ImGui::DockBuilderDockWindow("Scene stats", bottom);
        ImGui::DockBuilderDockWindow("Frame timings", bottom);
        ImGui::DockBuilderDockWindow("Lighting", bottom);

        ImGui::DockBuilderFinish(dockspace_id);
    }

    ImGui::DockSpace(dockspace_id, ImVec2(0.0F, 0.0F), ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::End();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    widget("Viewport", [&] {
        viewport_hovered = ImGui::IsWindowHovered();
        viewport_screen_pos = ImGui::GetCursorScreenPos();
        viewport_content_size = ImGui::GetContentRegionAvail();

        auto const target = renderer->viewport_target(frame_index);
        bool const has_room = viewport_content_size.x > 0.0F && viewport_content_size.y > 0.0F;
        if (target.valid() && has_room) {
            ImGui::Image(gui::linear_source_texture_id(target.index), viewport_content_size);
        }

        // Drawn into the Viewport window's own draw list so it always sits on top of the image.
        if (!is_playing && has_room) {
            auto &registry = active_scene()->get_registry();

            // The gizmo sits on the primary entity; the rest of the selection follows its change.
            auto const selected_entity = selection_context().primary();

            if (selected_entity != entt::null && registry.valid(selected_entity) &&
                registry.all_of<Components::Transform>(selected_entity)) {
                ImGuizmo::SetOrthographic(false);
                ImGuizmo::SetDrawlist();
                ImGuizmo::SetRect(viewport_screen_pos.x, viewport_screen_pos.y, viewport_content_size.x,
                                  viewport_content_size.y);

                auto const aspect =
                        viewport_content_size.y > 0.0F ? viewport_content_size.x / viewport_content_size.y : 1.0F;
                auto const view = camera.view();
                auto const projection = camera.projection(aspect);

                auto const previous_matrix = registry.get<Components::Transform>(selected_entity).matrix();
                auto matrix = previous_matrix;

                if (ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(projection), gizmo_operation, gizmo_mode,
                                         glm::value_ptr(matrix))) {
                    // Applied to each entity's own Transform, like the primary's, so the selection moves, turns and
                    // scales as a group. Descendants of selected entities follow their parent instead.
                    auto const delta = matrix * glm::inverse(previous_matrix);
                    auto const snapshot = selection_context().snapshot();

                    for (auto const entity: selection_roots(registry, snapshot.entities)) {
                        if (entity == selected_entity) {
                            set_transform_from_matrix(registry, entity, matrix);
                        } else if (registry.all_of<Components::Transform>(entity)) {
                            set_transform_from_matrix(registry, entity,
                                                      delta * registry.get<Components::Transform>(entity).matrix());
                        }
                    }

                    renderer->mark_dynamic_shadow_casters_dirty();
                }
            }
        }
    });
    ImGui::PopStyleVar();

    if (game) {
        game->on_ui(*active_scene(), *renderer);
    }
#if MINGW_VULKAN_TRACK_MEMORY
    widget("Memory", [] { on_memory_ui(); });
#endif
    widget("Console", [&] { terminal_widget.draw(); });

    widget("Load Model", [&] {
        ImGui::TextUnformatted("glTF / GLB model");

        ImGui::BeginDisabled(model_browser.is_open());
        if (ImGui::Button("Browse...")) {
            model_browse_target = ModelBrowseTarget::spawn_entity;
            model_browser.open("Load Model", model_file_filters());
        }
        ImGui::EndDisabled();

        // Newest first.
        for (auto const &load: std::views::reverse(model_loads)) {
            if (!load.settled) {
                ImGui::TextDisabled("Loading '%s'...", load.file_name.c_str());
            } else {
                ImGui::TextUnformatted(load.status.c_str());
            }
        }
    });

    widget("Assets", [&] {
        auto &assets = renderer->assets();

        // Recursively lists files under `root` with a lowercased extension in `extensions`. Missing or unreadable
        // directories yield nothing.
        auto scan_directory = [](std::filesystem::path const &root, std::span<std::string_view const> extensions) {
            std::vector<std::filesystem::path> found;
            std::error_code ec;

            if (!std::filesystem::exists(root, ec)) {
                return found;
            }

            for (auto const &entry: std::filesystem::recursive_directory_iterator(root, ec)) {
                if (!entry.is_regular_file()) {
                    continue;
                }

                auto extension = entry.path().extension().string();
                std::ranges::transform(extension, extension.begin(),
                                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

                if (std::ranges::find(extensions, extension) != extensions.end()) {
                    found.push_back(entry.path());
                }
            }

            return found;
        };

        auto const draw_bullet_entry = [](auto const &entry) {
            ImGui::BulletText("%s%s", entry.name.c_str(), entry.handle.valid() ? "" : " (invalid)");
        };

        // Registered entries, plus files under `root` that aren't loaded yet.
        auto draw_file_backed_section = [&]<typename HandleT>(char const *label, std::filesystem::path const &root,
                                                              std::span<std::string_view const> extensions,
                                                              NamedAssetTable<HandleT> &table, auto &&load,
                                                              auto &&draw_entry) {
            if (!ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen)) {
                return;
            }

            for (auto const &entry: table.entries()) {
                draw_entry(entry);
            }

            for (auto const &path: scan_directory(root, extensions)) {
                auto const filename = path.filename().string();
                if (table.find(filename).valid()) {
                    continue;
                }

                ImGui::PushID(path.string().c_str());
                if (ImGui::Button("Load")) {
                    load(path, filename);
                }
                ImGui::PopID();
                ImGui::SameLine();
                ImGui::TextDisabled("%s (not loaded)", path.string().c_str());
            }
        };

        static constexpr std::array<std::string_view, 2> model_extensions{".gltf", ".glb"};
        draw_file_backed_section(
                "Models", "assets/models", model_extensions, assets.models(),
                [&](std::filesystem::path const &path, std::string const &name) {
                    static_cast<void>(renderer->model_streamer().request(*renderer, path, engine_models.cube, name));
                },
                draw_bullet_entry);

        // ImageHandle::index doubles as the ImTextureID.
        auto const draw_texture_entry = [](auto const &entry) {
            if (!entry.handle.valid()) {
                ImGui::BulletText("%s (invalid)", entry.name.c_str());
                return;
            }

            float const thumb_size = ImGui::GetFontSize() * 2.0F;
            ImVec2 const row_start = ImGui::GetCursorPos();
            ImGui::Image(ImTextureID{entry.handle.index}, ImVec2(thumb_size, thumb_size));
            ImGui::SameLine();
            ImGui::SetCursorPosY(row_start.y + (thumb_size - ImGui::GetTextLineHeight()) * 0.5F);
            ImGui::TextUnformatted(entry.name.c_str());
        };

        static constexpr std::array<std::string_view, 3> texture_extensions{".png", ".jpg", ".jpeg"};
        draw_file_backed_section(
                "Textures", "assets/textures", texture_extensions, assets.textures(),
                [&](std::filesystem::path const &path, std::string const &name) {
                    static_cast<void>(renderer->request_texture(path, TextureRole::colour,
                                                                renderer->image_storage().white(), name));
                },
                draw_texture_entry);

        // Scripts are registered from C++, so this is a read-only list.
        auto draw_named_only_section = [&](char const *label, auto &table) {
            if (!ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen)) {
                return;
            }

            if (table.entries().empty()) {
                ImGui::TextDisabled("Nothing registered yet.");
                return;
            }

            for (auto const &entry: table.entries()) {
                ImGui::BulletText("%s%s", entry.name.c_str(), entry.handle.valid() ? "" : " (invalid)");
            }
        };

        // Shared by the "New Material" popup and each material's inline editor. Returns whether anything changed.
        auto const draw_material_fields = [&](MaterialCreateInfo &info) -> bool {
            bool changed = false;

            changed |= ImGui::ColorEdit4("Base colour", &info.base_colour_factor.x);
            changed |= ImGui::ColorEdit3("Emissive", &info.emissive_factor.x);
            changed |= ImGui::DragFloat("Emissive strength", &info.emissive_strength, 0.05F, 0.0F, 100.0F);
            changed |= ImGui::SliderFloat("Metallic", &info.metallic_factor, 0.0F, 1.0F);
            changed |= ImGui::SliderFloat("Roughness", &info.roughness_factor, 0.0F, 1.0F);
            changed |= ImGui::DragFloat("Normal scale", &info.normal_scale, 0.01F, 0.0F, 4.0F);
            changed |= ImGui::SliderFloat("Occlusion strength", &info.occlusion_strength, 0.0F, 1.0F);
            changed |= ImGui::DragFloat("Wind strength", &info.wind_strength, 0.01F, 0.0F, 4.0F, "%.2f",
                                        ImGuiSliderFlags_AlwaysClamp);

            int alpha_mode_index = static_cast<int>(info.alpha_mode);
            constexpr std::array<char const *, 3> alpha_mode_names{"Opaque", "Mask", "Blend"};
            if (ImGui::Combo("Alpha mode", &alpha_mode_index, alpha_mode_names.data(),
                             static_cast<int>(alpha_mode_names.size()))) {
                info.alpha_mode = static_cast<AlphaMode>(alpha_mode_index);
                changed = true;
            }
            if (info.alpha_mode == AlphaMode::mask) {
                changed |= ImGui::SliderFloat("Alpha cutoff", &info.alpha_cutoff, 0.0F, 1.0F);
            }

            bool casts_shadows = info.max_shadow_cascade != GpuMaterial::no_shadow_cascade;
            if (ImGui::Checkbox("Casts shadows", &casts_shadows)) {
                info.max_shadow_cascade = casts_shadows ? shadow_cascade_count - 1 : GpuMaterial::no_shadow_cascade;
                changed = true;
            }

            changed |= ImGui::Checkbox("Debug meshlet colours", &info.debug_meshlet_colours);

            // "(default)" means the slot's engine fallback image.
            auto const texture_picker = [&](char const *label, ImageHandle &slot, ImageHandle default_handle) {
                auto const &textures = assets.textures();
                bool const is_default = slot == default_handle;
                auto const current_name = textures.name_of(slot);
                std::string const preview =
                        is_default ? "(default)" : (current_name.empty() ? "(unnamed)" : std::string(current_name));

                if (!ImGui::BeginCombo(label, preview.c_str())) {
                    return;
                }

                if (ImGui::Selectable("(default)", is_default)) {
                    slot = default_handle;
                    changed = true;
                }
                for (auto const &entry: textures.entries()) {
                    bool const is_selected = entry.handle == slot;
                    if (ImGui::Selectable(entry.name.c_str(), is_selected)) {
                        slot = entry.handle;
                        changed = true;
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            };

            auto &images = renderer->image_storage();
            texture_picker("Base colour tex", info.base_colour_texture, images.white());
            texture_picker("Normal tex", info.normal_texture, images.flat_normal());
            texture_picker("Metallic/roughness tex", info.metallic_roughness_texture, images.metallic_roughness());
            texture_picker("Occlusion tex", info.occlusion_texture, images.occlusion());
            texture_picker("Emissive tex", info.emissive_texture, images.emissive());

            return changed;
        };

        if (ImGui::CollapsingHeader("Materials", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("New Material")) {
                new_material_info = MaterialCreateInfo{
                        .base_colour_texture = renderer->image_storage().white(),
                        .normal_texture = renderer->image_storage().flat_normal(),
                        .metallic_roughness_texture = renderer->image_storage().metallic_roughness(),
                        .occlusion_texture = renderer->image_storage().occlusion(),
                        .emissive_texture = renderer->image_storage().emissive(),
                        .sampler = renderer->sampler_storage().linear_repeat(),
                };
                new_material_name.clear();
                ImGui::OpenPopup("new_material_popup");
            }

            if (ImGui::BeginPopup("new_material_popup")) {
                std::array<char, 64> name_buf{};
                auto const copy_len = std::min(new_material_name.size(), name_buf.size() - 1);
                std::copy_n(new_material_name.begin(), copy_len, name_buf.begin());
                ImGui::SetNextItemWidth(ImGui::GetFontSize() * 16.0F);
                if (ImGui::InputTextWithHint("Name", "material name", name_buf.data(), name_buf.size())) {
                    new_material_name.assign(name_buf.data());
                }

                ImGui::Separator();
                draw_material_fields(new_material_info);
                ImGui::Separator();

                bool const name_taken =
                        !new_material_name.empty() && assets.materials().find(new_material_name).valid();
                if (name_taken) {
                    ImGui::TextColored(ImVec4(0.95F, 0.45F, 0.35F, 1.0F), "That name is already registered.");
                }

                ImGui::BeginDisabled(new_material_name.empty() || name_taken);
                if (ImGui::Button("Create")) {
                    auto const created = renderer->create_material(new_material_info, new_material_name);
                    if (!created) {
                        warn("Assets panel: failed to create material '{}': {}", new_material_name,
                             describe(created.error()));
                    }
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                if (ImGui::Button("Cancel")) {
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }

            auto &materials = assets.materials();
            bool any_shown = false;

            for (auto const &entry: materials.entries()) {
                if (is_material_pending_deletion(pending_deletions, entry.name)) {
                    continue; // Shown under "Recently deleted" instead.
                }
                any_shown = true;

                ImGui::PushID(entry.name.c_str());

                if (!entry.handle.valid()) {
                    ImGui::BulletText("%s (invalid)", entry.name.c_str());
                    ImGui::PopID();
                    continue;
                }

                bool const is_default = entry.handle == renderer->default_material();
                if (ImGui::TreeNode(entry.name.c_str())) {
                    if (auto const *info = renderer->material_storage().create_info(entry.handle)) {
                        auto edited = *info;
                        if (draw_material_fields(edited)) {
                            static_cast<void>(renderer->update_material(entry.handle, edited));
                        }
                    } else {
                        ImGui::TextDisabled("(material data unavailable)");
                    }

                    if (is_default) {
                        ImGui::TextDisabled("The default material can't be deleted.");
                    } else if (ImGui::Button("Delete")) {
                        // Destroyed only once the grace period elapses; until then the material stays fully usable.
                        pending_deletions.push_back(PendingDeletion{
                                .label = material_deletion_label(entry.name),
                                .delete_at = elapsed_time + deletion_grace_seconds,
                                .commit =
                                        [renderer = renderer.get(), handle = entry.handle, name = entry.name] {
                                            if (auto const result = renderer->destroy_material(handle); !result) {
                                                warn("Assets panel: failed to delete material '{}': {}", name,
                                                     describe(result.error()));
                                            }
                                        },
                        });
                    }

                    ImGui::TreePop();
                }

                ImGui::PopID();
            }

            if (!any_shown) {
                ImGui::TextDisabled("Nothing registered yet.");
            }

            static constexpr std::string_view material_prefix = "material:";
            bool has_material_deletions = std::ranges::any_of(pending_deletions, [&](PendingDeletion const &pending) {
                return pending.label.starts_with(material_prefix);
            });

            if (has_material_deletions && ImGui::TreeNode("Recently deleted")) {
                std::optional<std::size_t> restore_index;
                std::optional<std::size_t> commit_now_index;

                for (std::size_t index = 0; index < pending_deletions.size(); ++index) {
                    auto const &pending = pending_deletions[index];
                    if (!pending.label.starts_with(material_prefix)) {
                        continue;
                    }

                    ImGui::PushID(static_cast<int>(index));
                    auto const name = pending.label.substr(material_prefix.size());
                    auto const seconds_left = std::max(0.0F, pending.delete_at - elapsed_time);
                    ImGui::Text("%s -- deleting in %.0fs", name.c_str(), seconds_left);
                    ImGui::SameLine();
                    if (ImGui::SmallButton("Restore")) {
                        restore_index = index;
                    }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("Delete now")) {
                        commit_now_index = index;
                    }
                    ImGui::PopID();
                }

                // Both are indices into the same vector, so only one is applied per frame.
                if (restore_index) {
                    pending_deletions.erase(pending_deletions.begin() + static_cast<std::ptrdiff_t>(*restore_index));
                } else if (commit_now_index) {
                    pending_deletions[*commit_now_index].commit();
                    pending_deletions.erase(pending_deletions.begin() + static_cast<std::ptrdiff_t>(*commit_now_index));
                }

                ImGui::TreePop();
            }
        }

        draw_named_only_section("Scripts", assets.scripts());
    });

    widget("Hierarchy", [&] {
        auto const &style = ImGui::GetStyle();

        ImGui::SeparatorText("Gizmo");

        float const tool_icon_size = ImGui::GetFontSize() * 1.3F;

        auto const tool_button = [&](gui::EditorIcon icon, bool active, char const *id, char const *tooltip) {
            ImVec4 const bg = active ? ImVec4(0.26F, 0.59F, 0.98F, 0.45F) : ImVec4(0.0F, 0.0F, 0.0F, 0.0F);
            ImVec4 const tint = active ? ImVec4(1.0F, 1.0F, 1.0F, 1.0F) : ImVec4(0.68F, 0.68F, 0.72F, 1.0F);
            bool const pressed =
                    ImGui::ImageButton(id, editor_icons->texture(icon), ImVec2(tool_icon_size, tool_icon_size),
                                       ImVec2(0, 0), ImVec2(1, 1), bg, tint);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", tooltip);
            }
            return pressed;
        };

        if (tool_button(gui::EditorIcon::move, gizmo_operation == ImGuizmo::TRANSLATE, "##gizmo_move",
                        "Translate (1)")) {
            gizmo_operation = ImGuizmo::TRANSLATE;
        }
        ImGui::SameLine();
        if (tool_button(gui::EditorIcon::rotate, gizmo_operation == ImGuizmo::ROTATE, "##gizmo_rotate", "Rotate (2)")) {
            gizmo_operation = ImGuizmo::ROTATE;
        }
        ImGui::SameLine();
        if (tool_button(gui::EditorIcon::scale, gizmo_operation == ImGuizmo::SCALE, "##gizmo_scale", "Scale (3)")) {
            gizmo_operation = ImGuizmo::SCALE;
        }

        ImGui::SameLine(0.0F, style.ItemSpacing.x * 2.0F);

        bool const is_local = gizmo_mode == ImGuizmo::LOCAL;
        if (tool_button(is_local ? gui::EditorIcon::local : gui::EditorIcon::world, true, "##gizmo_space",
                        is_local ? "Local space (4) -- click for world space"
                                 : "World space (4) -- click for local space")) {
            gizmo_mode = is_local ? ImGuizmo::WORLD : ImGuizmo::LOCAL;
        }

        ImGui::SeparatorText("Entities");

        auto &registry = active_scene()->get_registry();

        auto const to_lower = [](std::string_view s) {
            std::string out(s);
            std::ranges::transform(out, out.begin(),
                                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return out;
        };
        auto const search_lower = to_lower(hierarchy_search);
        auto const matches_filter = [&](char const *name) {
            if (search_lower.empty()) {
                return true;
            }
            if (name == nullptr) {
                return false;
            }
            return to_lower(name).find(search_lower) != std::string::npos;
        };

        {
            float const icon_sz = ImGui::GetTextLineHeight();
            float const frame_h = ImGui::GetFrameHeight();
            ImVec2 const row_start = ImGui::GetCursorPos();

            ImGui::SetCursorPos(ImVec2(row_start.x + style.FramePadding.x, row_start.y + (frame_h - icon_sz) * 0.5F));
            ImGui::ImageWithBg(editor_icons->texture(gui::EditorIcon::search), ImVec2(icon_sz, icon_sz), ImVec2(0, 0),
                               ImVec2(1, 1), ImVec4(0, 0, 0, 0), ImVec4(0.55F, 0.55F, 0.60F, 1.0F));

            ImGui::SetCursorPos(
                    ImVec2(row_start.x + icon_sz + style.FramePadding.x + style.ItemInnerSpacing.x, row_start.y));
            ImGui::SetNextItemWidth(-1.0F);

            std::array<char, 128> search_buf{};
            auto const copy_len = std::min(hierarchy_search.size(), search_buf.size() - 1);
            std::copy_n(hierarchy_search.begin(), copy_len, search_buf.begin());
            if (ImGui::InputTextWithHint("##hierarchy_search", "Search entities...", search_buf.data(),
                                         search_buf.size())) {
                hierarchy_search.assign(search_buf.data());
            }
        }

        struct EntityVisual {
            gui::EditorIcon icon;
            ImVec4 tint;
        };

        auto const visual_for = [&](entt::entity entity) -> EntityVisual {
            if (registry.all_of<Components::PointLight>(entity)) {
                return {.icon = gui::EditorIcon::point_light, .tint = ImVec4(1.00F, 0.84F, 0.35F, 1.0F)};
            }
            if (registry.all_of<Components::SpotLight>(entity)) {
                return {.icon = gui::EditorIcon::spot_light, .tint = ImVec4(1.00F, 0.84F, 0.35F, 1.0F)};
            }
            if (registry.all_of<Components::PlayerTag>(entity)) {
                return {.icon = gui::EditorIcon::player, .tint = ImVec4(0.47F, 0.86F, 0.55F, 1.0F)};
            }
            if (registry.all_of<Components::BulletTag>(entity)) {
                return {.icon = gui::EditorIcon::bullet, .tint = ImVec4(1.00F, 0.53F, 0.38F, 1.0F)};
            }
            if (registry.all_of<Components::Model>(entity) || registry.all_of<Components::InstancedModel>(entity)) {
                return {.icon = gui::EditorIcon::mesh, .tint = ImVec4(0.88F, 0.64F, 0.37F, 1.0F)};
            }
            if (registry.all_of<Components::Script>(entity)) {
                return {.icon = gui::EditorIcon::script, .tint = ImVec4(0.42F, 0.70F, 1.00F, 1.0F)};
            }
            return {.icon = gui::EditorIcon::empty, .tint = ImVec4(0.60F, 0.60F, 0.64F, 1.0F)};
        };

        // Bullets are grouped under one collapsible node since they spawn in bursts.
        auto const bullet_view =
                registry.view<Components::Transform, Components::GeneratedMeta, Components::BulletTag>();
        auto const bullet_count = static_cast<std::uint32_t>(std::distance(bullet_view.begin(), bullet_view.end()));

        auto const meta_view =
                registry.view<Components::Transform, Components::Meta>(entt::exclude<Components::BulletTag>);
        auto const generated_view =
                registry.view<Components::Transform, Components::GeneratedMeta>(entt::exclude<Components::BulletTag>);

        // Dedup so an entity carrying both Meta and GeneratedMeta isn't listed twice (duplicate ImGui ID).
        std::vector<entt::entity> listed_entities;
        listed_entities.reserve(static_cast<std::size_t>(std::distance(generated_view.begin(), generated_view.end())) +
                                static_cast<std::size_t>(std::distance(meta_view.begin(), meta_view.end())));
        std::unordered_set<entt::entity> listed_set;

        for (auto const entity: generated_view) {
            listed_entities.push_back(entity);
            listed_set.insert(entity);
        }
        for (auto const entity: meta_view) {
            if (listed_set.insert(entity).second) {
                listed_entities.push_back(entity);
            }
        }

        std::unordered_map<entt::entity, std::vector<entt::entity>> children_of;
        std::vector<entt::entity> root_entities;
        root_entities.reserve(listed_entities.size());

        for (auto const entity: listed_entities) {
            auto const *parent = registry.try_get<Components::Parent>(entity);
            bool const has_listed_parent = parent != nullptr && registry.valid(parent->entity) &&
                                           registry.any_of<Components::Meta, Components::GeneratedMeta>(parent->entity);
            if (has_listed_parent) {
                children_of[parent->entity].push_back(entity);
            } else {
                root_entities.push_back(entity);
            }
        }

        // A parent stays visible while filtering if it or any descendant matches.
        std::unordered_map<entt::entity, bool> match_cache;
        std::function<bool(entt::entity)> subtree_matches = [&](entt::entity entity) -> bool {
            if (auto const cached = match_cache.find(entity); cached != match_cache.end()) {
                return cached->second;
            }
            // Seeded before recursing so a cyclic Parent chain terminates.
            match_cache[entity] = true;

            bool result = matches_filter(entity_display_name(registry, entity).c_str());
            if (!result) {
                if (auto const it = children_of.find(entity); it != children_of.end()) {
                    for (auto const child: it->second) {
                        if (subtree_matches(child)) {
                            result = true;
                            break;
                        }
                    }
                }
            }

            match_cache[entity] = result;
            return result;
        };

        // Actions picked from the context menus are applied after the tree is drawn, so the registry isn't mutated
        // while the views are being iterated.
        enum class HierarchyAction : std::uint8_t { none, add_child, duplicate, remove, reparent };
        HierarchyAction pending_action = HierarchyAction::none;
        // add_child: the new entity's parent (entt::null = root).
        // duplicate/remove: unused; they act on the selection.
        // reparent: the new parent (entt::null = root); the moved entity is drag_reparent_source.
        entt::entity action_target = entt::null;
        entt::entity drag_reparent_source = entt::null;

        std::size_t row_index = 0;

        auto &selection = selection_context();
        // The selection can outlive its entities (deleted, or the registry swapped by play/stop).
        selection.retain_if([&](entt::entity entity) { return registry.valid(entity); });

        // Rows in draw order. ImGui's multi-select identifies each row by its index here, and reports shift-click
        // ranges in those indices.
        std::vector<entt::entity> drawn_rows;
        drawn_rows.reserve(listed_entities.size() + bullet_count);
        // Row clicked this frame; becomes the primary entity so the Inspector follows the click.
        entt::entity clicked_entity = entt::null;

        auto const begin_rename = [&](entt::entity entity) {
            renaming_entity = entity;
            copy_to_buffer(rename_buffer, entity_display_name(registry, entity));
            rename_needs_focus = true;
        };

        // Draws one row, plus a tree node with the entity's children if it has any.
        std::function<void(entt::entity)> draw_entity_node = [&](entt::entity entity) {
            if (!subtree_matches(entity)) {
                return;
            }

            auto const name = entity_display_name(registry, entity);
            auto const child_it = children_of.find(entity);
            bool const has_children = child_it != children_of.end() && !child_it->second.empty();

            auto visual = visual_for(entity);
            if (has_children && visual.icon == gui::EditorIcon::empty) {
                // Pure grouping entity.
                visual = {.icon = gui::EditorIcon::folder, .tint = ImVec4(0.95F, 0.80F, 0.45F, 1.0F)};
            }

            ImGui::PushID(static_cast<int>(entity));

            float const row_height = ImGui::GetFrameHeight();
            float const icon_size = ImGui::GetTextLineHeight();
            ImVec2 const row_pos = ImGui::GetCursorPos();
            ImVec2 const row_screen_pos = ImGui::GetCursorScreenPos();
            float const avail_width = ImGui::GetContentRegionAvail().x;

            if (row_index++ % 2 == 1) {
                ImGui::GetWindowDrawList()->AddRectFilled(
                        row_screen_pos, ImVec2(row_screen_pos.x + avail_width, row_screen_pos.y + row_height),
                        ImGui::GetColorU32(ImGuiCol_TableRowBgAlt));
            }

            bool const is_selected = selection.contains(entity);
            float label_x = row_pos.x + style.FramePadding.x;
            ImVec2 next_row_pos;
            bool open = false;

            if (has_children) {
                ImGuiTreeNodeFlags const flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnArrow |
                                                 ImGuiTreeNodeFlags_FramePadding |
                                                 (is_selected ? ImGuiTreeNodeFlags_Selected : ImGuiTreeNodeFlags_None);
                // The leading space and FramePadding give the node the same height as the Selectable rows.
                // OpenOnArrow: clicking the row selects, only the arrow expands.
                ImGui::SetNextItemSelectionUserData(static_cast<ImGuiSelectionUserData>(drawn_rows.size()));
                drawn_rows.push_back(entity);
                open = ImGui::TreeNodeEx(" ##node", flags);
                next_row_pos = ImGui::GetCursorPos();
                label_x = row_pos.x + ImGui::GetTreeNodeToLabelSpacing();
            } else {
                // Selection changes arrive as multi-select requests, so the return value isn't needed.
                ImGui::SetNextItemSelectionUserData(static_cast<ImGuiSelectionUserData>(drawn_rows.size()));
                drawn_rows.push_back(entity);
                static_cast<void>(ImGui::Selectable("##row", is_selected, ImGuiSelectableFlags_None,
                                                    ImVec2(avail_width, row_height)));
                next_row_pos = ImGui::GetCursorPos();
            }

            if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                clicked_entity = entity;
            }
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                begin_rename(entity);
            }

            // Multi-select already selects an unselected row on right-click, so the menu acts on the selection.
            ImGui::OpenPopupOnItemClick("entity_context", ImGuiPopupFlags_MouseButtonRight);

            if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
                ImGui::SetDragDropPayload("HIERARCHY_ENTITY", &entity, sizeof(entity));
                if (auto const count = selection.size(); is_selected && count > 1) {
                    ImGui::Text("%zu entities", count);
                } else {
                    ImGui::TextUnformatted(name.c_str());
                }
                ImGui::EndDragDropSource();
            }

            if (ImGui::BeginDragDropTarget()) {
                if (auto const *payload = ImGui::AcceptDragDropPayload("HIERARCHY_ENTITY")) {
                    entt::entity dropped{};
                    std::memcpy(&dropped, payload->Data, sizeof(dropped));
                    pending_action = HierarchyAction::reparent;
                    action_target = entity;
                    drag_reparent_source = dropped;
                }
                ImGui::EndDragDropTarget();
            }

            // Icon and label are drawn over the widget above, then the cursor is restored.
            ImGui::SetCursorPos(ImVec2(label_x, row_pos.y + (row_height - icon_size) * 0.5F));
            ImGui::ImageWithBg(editor_icons->texture(visual.icon), ImVec2(icon_size, icon_size), ImVec2(0, 0),
                               ImVec2(1, 1), ImVec4(0, 0, 0, 0), visual.tint);

            float const text_x = label_x + icon_size + style.ItemInnerSpacing.x;
            if (renaming_entity == entity) {
                ImGui::SetCursorPos(ImVec2(text_x, row_pos.y));
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (rename_needs_focus) {
                    ImGui::SetKeyboardFocusHere();
                    rename_needs_focus = false;
                }

                bool const submitted =
                        ImGui::InputText("##rename", rename_buffer.data(), rename_buffer.size(),
                                         ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
                // Enter or clicking away commits; Escape cancels.
                if (submitted || (ImGui::IsItemDeactivated() && !ImGui::IsKeyPressed(ImGuiKey_Escape))) {
                    rename_entity(registry, entity, rename_buffer.data());
                    renaming_entity = entt::null;
                } else if (ImGui::IsItemDeactivated()) {
                    renaming_entity = entt::null;
                }
            } else {
                ImGui::SetCursorPos(ImVec2(text_x, row_pos.y + (row_height - ImGui::GetTextLineHeight()) * 0.5F));
                ImGui::TextUnformatted(name.c_str());
            }

            ImGui::SetCursorPos(next_row_pos);

            if (has_children && open) {
                for (auto const child: child_it->second) {
                    draw_entity_node(child);
                }
                ImGui::TreePop();
            }

            if (ImGui::BeginPopup("entity_context")) {
                auto const count = selection.size();

                if (ImGui::MenuItem("Add Child Entity")) {
                    pending_action = HierarchyAction::add_child;
                    action_target = entity;
                }
                if (ImGui::MenuItem("Rename", "F2")) {
                    begin_rename(entity);
                }
                // Duplicate and Delete act on the whole selection, which includes this row.
                if (ImGui::MenuItem(count > 1 ? "Duplicate Selected" : "Duplicate", "Ctrl+D")) {
                    pending_action = HierarchyAction::duplicate;
                }
                ImGui::Separator();
                if (ImGui::MenuItem(count > 1 ? "Delete Selected" : "Delete", "Del")) {
                    pending_action = HierarchyAction::remove;
                }
                ImGui::EndPopup();
            }

            ImGui::PopID();
        };

        auto const total_count = bullet_count + static_cast<std::uint32_t>(listed_entities.size());
        if (auto const selected_count = selection.size(); selected_count > 1) {
            ImGui::TextDisabled("%u %s, %zu selected", total_count, total_count == 1 ? "entity" : "entities",
                                selected_count);
        } else {
            ImGui::TextDisabled("%u %s", total_count, total_count == 1 ? "entity" : "entities");
        }

        // Applies one BeginMultiSelect()/EndMultiSelect() batch as a single selection change.
        auto const apply_selection_requests = [&](ImGuiMultiSelectIO const *io) {
            if (io == nullptr || io->Requests.empty()) {
                return;
            }
            // Focusing the rename field moves keyboard nav onto an item that isn't selectable, which multi-select
            // answers by clearing the selection. Only a click changes the selection while a row is being renamed.
            if (renaming_entity != entt::null && !ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                return;
            }

            selection.modify([&](SelectionContext::Transaction &transaction) {
                for (auto const &request: io->Requests) {
                    if (request.Type == ImGuiSelectionRequestType_SetAll) {
                        transaction.clear();
                        if (!request.Selected) {
                            continue;
                        }
                        // Ctrl+A: everything the filter shows, collapsed children and bullets included.
                        for (auto const entity: listed_entities) {
                            if (matches_filter(entity_display_name(registry, entity).c_str())) {
                                transaction.set(entity, true);
                            }
                        }
                        for (auto const entity: bullet_view) {
                            if (matches_filter(registry.get<Components::GeneratedMeta>(entity).name.c_str())) {
                                transaction.set(entity, true);
                            }
                        }
                    } else if (request.Type == ImGuiSelectionRequestType_SetRange) {
                        auto const first = std::max<ImGuiSelectionUserData>(
                                std::min(request.RangeFirstItem, request.RangeLastItem), 0);
                        auto const last = std::min<ImGuiSelectionUserData>(
                                std::max(request.RangeFirstItem, request.RangeLastItem),
                                static_cast<ImGuiSelectionUserData>(drawn_rows.size()) - 1);
                        for (auto row = first; row <= last; ++row) {
                            transaction.set(drawn_rows[static_cast<std::size_t>(row)], request.Selected);
                        }
                    }
                }
            });
        };

        constexpr ImGuiMultiSelectFlags multi_select_flags = ImGuiMultiSelectFlags_ClearOnEscape |
                                                             ImGuiMultiSelectFlags_ClearOnClickVoid |
                                                             ImGuiMultiSelectFlags_BoxSelect1d;
        apply_selection_requests(ImGui::BeginMultiSelect(multi_select_flags, static_cast<int>(selection.size()),
                                                         static_cast<int>(total_count)));

        bool const any_bullet_matches =
                search_lower.empty() || std::ranges::any_of(bullet_view, [&](entt::entity e) {
                    return matches_filter(registry.get<Components::GeneratedMeta>(e).name.c_str());
                });

        if (bullet_count > 0 && any_bullet_matches) {
            ImGui::PushID("bullets_group");

            float const row_height = ImGui::GetFrameHeight();
            float const icon_size = ImGui::GetTextLineHeight();
            ImVec2 const row_pos = ImGui::GetCursorPos();

            // See draw_entity_node for the leading space and FramePadding.
            bool const open = ImGui::TreeNodeEx(" ##bullets_node",
                                                ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_FramePadding);
            ImVec2 const after_tree_pos = ImGui::GetCursorPos();

            float const label_x = row_pos.x + ImGui::GetTreeNodeToLabelSpacing();
            ImGui::SetCursorPos(ImVec2(label_x, row_pos.y + (row_height - icon_size) * 0.5F));
            ImGui::ImageWithBg(editor_icons->texture(gui::EditorIcon::folder), ImVec2(icon_size, icon_size),
                               ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0), ImVec4(0.95F, 0.80F, 0.45F, 1.0F));
            ImGui::SetCursorPos(ImVec2(label_x + icon_size + style.ItemInnerSpacing.x,
                                       row_pos.y + (row_height - ImGui::GetTextLineHeight()) * 0.5F));
            ImGui::Text("Bullets (%u)", bullet_count);

            ImGui::SetCursorPos(after_tree_pos);

            if (open) {
                for (auto const entity: bullet_view) {
                    draw_entity_node(entity);
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }

        for (auto const entity: root_entities) {
            draw_entity_node(entity);
        }

        // The rows end with SetCursorPos(); ImGui asserts unless a real item follows.
        ImGui::Dummy(ImVec2(0.0F, 0.0F));

        // Dropping below the tree moves the entity back to root.
        if (ImGui::BeginDragDropTarget()) {
            if (auto const *payload = ImGui::AcceptDragDropPayload("HIERARCHY_ENTITY")) {
                entt::entity dropped{};
                std::memcpy(&dropped, payload->Data, sizeof(dropped));
                pending_action = HierarchyAction::reparent;
                action_target = entt::null;
                drag_reparent_source = dropped;
            }
            ImGui::EndDragDropTarget();
        }

        apply_selection_requests(ImGui::EndMultiSelect());
        if (clicked_entity != entt::null) {
            selection.modify([&](SelectionContext::Transaction &transaction) {
                transaction.set_primary(clicked_entity);
            });
        }

        // Shortcuts while the Hierarchy is focused and no text field has the keyboard.
        if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && !ImGui::GetIO().WantTextInput) {
            if (auto const primary = selection.primary();
                ImGui::IsKeyPressed(ImGuiKey_F2, false) && primary != entt::null) {
                begin_rename(primary);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_Delete, false) && !selection.empty()) {
                pending_action = HierarchyAction::remove;
            }
            if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_D) && !selection.empty()) {
                pending_action = HierarchyAction::duplicate;
            }
        }


        // NoOpenOverItems: rows have their own "entity_context" popup.
        if (ImGui::BeginPopupContextWindow("hierarchy_bg_context",
                                           ImGuiPopupFlags_MouseButtonRight | ImGuiPopupFlags_NoOpenOverItems)) {
            if (ImGui::MenuItem("Create Empty Entity")) {
                pending_action = HierarchyAction::add_child;
                action_target = entt::null;
            }
            ImGui::EndPopup();
        }

        auto const selected_roots = [&] { return selection_roots(registry, selection.snapshot().entities); };

        switch (pending_action) {
            case HierarchyAction::add_child: {
                auto new_entity = GeneratedEntity{active_scene(), "Entity"};
                // Only entities with a Transform are listed.
                new_entity.emplace<Components::Transform>();
                if (action_target != entt::null && registry.valid(action_target)) {
                    new_entity.emplace<Components::Parent>(Components::Parent{.entity = action_target});
                }
                selection.select(new_entity);
                break;
            }
            case HierarchyAction::duplicate: {
                // Recreates `source`'s subtree under `parent` (entt::null = root).
                std::function<entt::entity(entt::entity, entt::entity)> duplicate_subtree =
                        [&](entt::entity source, entt::entity parent) -> entt::entity {
                    auto const clone = registry.create();

                    copy_components<Components::Transform, Components::Model, Components::InstancedModel,
                                    Components::RigidBody, Components::MaterialOverride, Components::PlayerTag,
                                    Components::Lifetime, Components::PointLight, Components::SpotLight,
                                    Components::Script, Components::BulletTag>(registry, source, clone);

                    // The clone takes its own reference, so deleting either one leaves the other's model alive.
                    if (auto const *model = registry.try_get<Components::Model>(clone);
                        model != nullptr && registry.all_of<Components::StreamedModelTag>(source)) {
                        renderer->retain_model(model->model);
                        registry.emplace<Components::StreamedModelTag>(clone);
                    }

                    // Duplicates are always named through GeneratedMeta.
                    registry.emplace<Components::GeneratedMeta>(
                            clone,
                            Components::GeneratedMeta{.name = entity_display_name(registry, source) + " (Copy)"});

                    if (parent != entt::null) {
                        registry.emplace<Components::Parent>(clone, Components::Parent{.entity = parent});
                    }

                    if (auto const it = children_of.find(source); it != children_of.end()) {
                        for (auto const child: it->second) {
                            duplicate_subtree(child, clone);
                        }
                    }

                    return clone;
                };

                // The copies replace the originals in the selection, so they can be moved straight away.
                std::vector<entt::entity> clones;
                for (auto const source: selected_roots()) {
                    auto const *source_parent = registry.try_get<Components::Parent>(source);
                    clones.push_back(
                            duplicate_subtree(source, source_parent != nullptr ? source_parent->entity : entt::null));
                }
                selection.assign(clones);
                break;
            }
            case HierarchyAction::remove: {
                std::function<void(entt::entity)> delete_subtree = [&](entt::entity target) {
                    if (!registry.valid(target)) {
                        return;
                    }
                    if (auto const it = children_of.find(target); it != children_of.end()) {
                        for (auto const child: it->second) {
                            delete_subtree(child);
                        }
                    }

                    // Only StreamedModelTag entities own a model reference that can be released.
                    if (auto const *model = registry.try_get<Components::Model>(target);
                        model != nullptr && registry.all_of<Components::StreamedModelTag>(target)) {
                        renderer->release_model(model->model);
                    }

                    registry.destroy(target);
                };

                for (auto const target: selected_roots()) {
                    delete_subtree(target);
                }
                selection.clear();
                break;
            }
            case HierarchyAction::reparent: {
                // Dragging a selected row moves the whole selection; an unselected row moves alone.
                auto const sources = selection.contains(drag_reparent_source)
                                             ? selected_roots()
                                             : std::vector<entt::entity>{drag_reparent_source};

                for (auto const source: sources) {
                    if (!registry.valid(source) || source == action_target) {
                        continue;
                    }

                    // Reject drops onto one of the dragged entity's own descendants.
                    bool creates_cycle = false;
                    for (auto walk = action_target; walk != entt::null && registry.valid(walk);) {
                        if (walk == source) {
                            creates_cycle = true;
                            break;
                        }
                        auto const *walk_parent = registry.try_get<Components::Parent>(walk);
                        walk = walk_parent != nullptr ? walk_parent->entity : entt::null;
                    }

                    if (creates_cycle) {
                        continue;
                    }

                    if (action_target == entt::null) {
                        registry.remove<Components::Parent>(source);
                    } else {
                        registry.emplace_or_replace<Components::Parent>(source,
                                                                        Components::Parent{.entity = action_target});
                    }
                }
                break;
            }
            case HierarchyAction::none:
                break;
        }

        if (renaming_entity != entt::null && !registry.valid(renaming_entity)) {
            renaming_entity = entt::null;
        }
    });

    widget("Inspector", [&] {
        auto &registry = active_scene()->get_registry();
        // Edits apply to the primary entity: the one selected last.
        auto const selected_entity = selection_context().primary();

        if (selected_entity == entt::null || !registry.valid(selected_entity)) {
            ImGui::TextDisabled("No entity selected");
            return;
        }

        if (auto const count = selection_context().size(); count > 1) {
            ImGui::TextDisabled("%zu entities selected -- editing the last one", count);
        }

        // Refilled from the entity whenever the field isn't being edited, so it follows the selection and renames
        // made elsewhere. An edit is committed to the entity it was typed for once the field loses focus.
        auto const name_id = ImGui::GetID("##entity_name");
        if (ImGui::GetActiveID() != name_id) {
            if (inspector_name_dirty) {
                rename_entity(registry, inspector_name_entity, inspector_name_buffer.data());
                inspector_name_dirty = false;
            }
            inspector_name_entity = selected_entity;
            copy_to_buffer(inspector_name_buffer, entity_display_name(registry, selected_entity));
        }
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::InputText("##entity_name", inspector_name_buffer.data(), inspector_name_buffer.size())) {
            inspector_name_dirty = true;
        }
        ImGui::Separator();

        // One collapsible section per present component. `draw_fields` edits in place and returns whether it changed,
        // which drives patch<T>. Closing the header queues the removal.
        auto section = [&]<typename T>(char const *label, auto &&draw_fields) {
            if (!registry.all_of<T>(selected_entity)) {
                return;
            }

            bool open = true;
            ImGui::PushID(label);
            if (ImGui::CollapsingHeader(label, &open, ImGuiTreeNodeFlags_DefaultOpen)) {
                if (draw_fields(registry.get<T>(selected_entity))) {
                    registry.patch<T>(selected_entity);

                    if constexpr (std::is_same_v<T, Components::Transform>) {
                        renderer->mark_dynamic_shadow_casters_dirty();
                    }
                }
            }
            ImGui::PopID();

            if (!open) {
                // An owned model's reference goes with the component.
                if constexpr (std::is_same_v<T, Components::Model>) {
                    if (registry.all_of<Components::StreamedModelTag>(selected_entity)) {
                        renderer->release_model(registry.get<Components::Model>(selected_entity).model);
                        registry.remove<Components::StreamedModelTag>(selected_entity);
                    }
                }
                registry.remove<T>(selected_entity);
            }
        };

        section.operator()<Components::Transform>("Transform", draw_transform);
        section.operator()<Components::PointLight>("Point Light", draw_point_light);
        section.operator()<Components::SpotLight>("Spot Light", draw_spot_light);
        section.operator()<Components::RigidBody>("Rigid Body", draw_rigid_body);
        section.operator()<Components::Lifetime>("Lifetime", draw_lifetime);

        section.operator()<Components::Model>("Model", [&](Components::Model &model) {
            ImGui::Text("Handle: index %u, generation %u (%s)", model.model.index, model.model.generation,
                        model.model.valid() ? "valid" : "invalid");

            if (auto const bounds = renderer->model_bounds(model.model)) {
                auto const &[min, max] = *bounds;
                ImGui::Text("Bounds (model space): min (%.2f, %.2f, %.2f)", min.x, min.y, min.z);
                ImGui::Text("                      max (%.2f, %.2f, %.2f)", max.x, max.y, max.z);
            } else {
                ImGui::TextDisabled("Bounds unavailable");
            }

            if (auto const submesh_bounds = renderer->model_submesh_bounds(model.model)) {
                ImGui::Text("Submeshes: %u", static_cast<std::uint32_t>(submesh_bounds->size()));
            }

            if (auto const lights = renderer->model_lights(model.model); !lights.empty()) {
                ImGui::Text("Embedded lights: %u", static_cast<std::uint32_t>(lights.size()));
            }

            if (auto const state = renderer->model_streamer().state(model.model); state == ModelRequestState::loading) {
                ImGui::TextDisabled("Loading -- showing the placeholder model");
            } else if (state == ModelRequestState::failed) {
                auto const reason = renderer->model_streamer().failure_reason(model.model);
                ImGui::TextColored(ImVec4(1.0F, 0.45F, 0.40F, 1.0F),
                                   "Load failed: %.*s -- showing the placeholder model", static_cast<int>(reason.size()),
                                   reason.data());
            }

            auto &models = renderer->assets().models();
            auto const current_name = models.name_of(model.model);

            // set_entity_model() patches the component itself, so the section's patch isn't needed.
            if (ImGui::BeginCombo("Asset", current_name.empty() ? "(unnamed)" : std::string(current_name).c_str())) {
                for (auto const &entry: models.entries()) {
                    bool const is_selected = entry.handle == model.model;
                    if (ImGui::Selectable(entry.name.c_str(), is_selected) && !is_selected && entry.handle.valid()) {
                        // The combo shares an already-owned handle, so the entity needs its own reference.
                        renderer->retain_model(entry.handle);
                        set_entity_model(registry, selected_entity, entry.handle);
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::SameLine();
            ImGui::BeginDisabled(model_browser.is_open());
            if (ImGui::Button("Browse...##model")) {
                model_browse_target = ModelBrowseTarget::inspector;
                model_browse_entity = selected_entity;
                model_browser.open("Change Model", model_file_filters());
            }
            ImGui::EndDisabled();

            return false;
        });
        section.operator()<Components::MaterialOverride>("Material Override", [&](Components::MaterialOverride &mat) {
            ImGui::Text("Handle: index %u, generation %u (%s)", mat.material.index, mat.material.generation,
                        mat.material.valid() ? "valid" : "invalid");

            auto &materials = renderer->assets().materials();
            if (materials.entries().empty()) {
                ImGui::TextDisabled("No named materials registered yet.");
                return false;
            }

            bool changed = false;
            auto const current_name = materials.name_of(mat.material);
            if (ImGui::BeginCombo("Asset", current_name.empty() ? "(unnamed)" : std::string(current_name).c_str())) {
                for (auto const &entry: materials.entries()) {
                    bool const is_selected = entry.handle == mat.material;
                    // Materials queued for deletion aren't offered as new picks.
                    if (!is_selected && is_material_pending_deletion(pending_deletions, entry.name)) {
                        continue;
                    }
                    if (ImGui::Selectable(entry.name.c_str(), is_selected) && entry.handle != mat.material) {
                        mat.material = entry.handle;
                        changed = true;
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            return changed;
        });
        section.operator()<Components::Script>("Script", [&](Components::Script &script) {
            ImGui::Text("Handle: index %u, generation %u (%s)", script.script.index, script.script.generation,
                        script.script.valid() ? "valid" : "invalid");

            auto &scripts = renderer->assets().scripts();
            if (scripts.entries().empty()) {
                ImGui::TextDisabled("No named scripts registered yet.");
                return false;
            }

            bool changed = false;
            auto const current_name = scripts.name_of(script.script);
            if (ImGui::BeginCombo("Asset", current_name.empty() ? "(unnamed)" : std::string(current_name).c_str())) {
                for (auto const &entry: scripts.entries()) {
                    bool const is_selected = entry.handle == script.script;
                    if (ImGui::Selectable(entry.name.c_str(), is_selected) && entry.handle != script.script) {
                        script.script = entry.handle;
                        changed = true;
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            return changed;
        });
        // Empty tag types have no storage, so get<T>() can't be passed to draw_fields.
        auto tag_section = [&]<typename T>(char const *label) {
            if (!registry.all_of<T>(selected_entity)) {
                return;
            }

            bool open = true;
            ImGui::PushID(label);
            if (ImGui::CollapsingHeader(label, &open, ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::TextDisabled("Marker component -- no fields.");
            }
            ImGui::PopID();

            if (!open) {
                registry.remove<T>(selected_entity);
            }
        };

        tag_section.operator()<Components::PlayerTag>("Player Tag");
        tag_section.operator()<Components::BulletTag>("Bullet Tag");

        ImGui::Separator();

        if (ImGui::Button("Add Component")) {
            ImGui::OpenPopup("inspector_add_component");
        }

        if (ImGui::BeginPopup("inspector_add_component")) {
            bool const has_transform = registry.all_of<Components::Transform>(selected_entity);

            // Lights and rigid bodies read the Transform, so it has to exist first.
            if (!has_transform) {
                if (ImGui::MenuItem("Transform")) {
                    registry.emplace<Components::Transform>(selected_entity);
                }
            } else {
                if (!registry.all_of<Components::PointLight>(selected_entity) && ImGui::MenuItem("Point Light")) {
                    registry.emplace<Components::PointLight>(selected_entity);
                }
                if (!registry.all_of<Components::SpotLight>(selected_entity) && ImGui::MenuItem("Spot Light")) {
                    registry.emplace<Components::SpotLight>(selected_entity);
                }
                if (!registry.all_of<Components::RigidBody>(selected_entity) && ImGui::MenuItem("Rigid Body")) {
                    // Takes effect the next time physics is populated (entering Play).
                    registry.emplace<Components::RigidBody>(selected_entity);
                }

                if (!registry.all_of<Components::Model>(selected_entity) &&
                    !renderer->assets().models().entries().empty() && ImGui::BeginMenu("Model")) {
                    for (auto const &entry: renderer->assets().models().entries()) {
                        if (ImGui::MenuItem(entry.name.c_str())) {
                            renderer->retain_model(entry.handle);
                            registry.emplace<Components::Model>(selected_entity,
                                                                Components::Model{.model = entry.handle});
                            registry.emplace<Components::StreamedModelTag>(selected_entity);
                        }
                    }
                    ImGui::EndMenu();
                }
            }

            if (!registry.all_of<Components::Lifetime>(selected_entity) && ImGui::MenuItem("Lifetime")) {
                registry.emplace<Components::Lifetime>(selected_entity,
                                                       Components::Lifetime{.remaining_seconds = 5.0F});
            }

            if (!registry.all_of<Components::MaterialOverride>(selected_entity) &&
                !renderer->assets().materials().entries().empty() && ImGui::BeginMenu("Material Override")) {
                for (auto const &entry: renderer->assets().materials().entries()) {
                    if (is_material_pending_deletion(pending_deletions, entry.name)) {
                        continue;
                    }
                    if (ImGui::MenuItem(entry.name.c_str())) {
                        registry.emplace<Components::MaterialOverride>(
                                selected_entity, Components::MaterialOverride{.material = entry.handle});
                    }
                }
                ImGui::EndMenu();
            }

            if (!registry.all_of<Components::Script>(selected_entity) &&
                !renderer->assets().scripts().entries().empty() && ImGui::BeginMenu("Script")) {
                for (auto const &entry: renderer->assets().scripts().entries()) {
                    if (ImGui::MenuItem(entry.name.c_str())) {
                        registry.emplace<Components::Script>(selected_entity,
                                                             Components::Script{.script = entry.handle});
                    }
                }
                ImGui::EndMenu();
            }

            ImGui::EndPopup();
        }
    });

    widget("Simulation", [&] {
        if (is_playing) {
            if (ImGui::Button("Stop")) {
                stop();
            }
        } else {
            if (ImGui::Button("Play")) {
                play();
            }
        }

        ImGui::SameLine();
        // Editable mid-play to switch an embedded session to fullscreen and back.
        ImGui::Checkbox("Fullscreen", &play_fullscreen);
    });

    widget("Scene stats", [&] {
        auto const &stats = renderer->last_frame_stats();
        auto const &pipeline_stats = renderer->last_frame_pipeline_stats();

        constexpr auto fmt = [](std::uint64_t count) {
            static thread_local std::array<char, 64> buf{};
            if (count >= 1'000'000'000) {
                std::snprintf(buf.data(), buf.size(), "%.2fB", static_cast<double>(count) / 1e9F);
            } else if (count >= 1'000'000) {
                std::snprintf(buf.data(), buf.size(), "%.2fM", static_cast<double>(count) / 1e6F);
            } else if (count >= 1'000) {
                std::snprintf(buf.data(), buf.size(), "%.2fK", static_cast<double>(count) / 1e3F);
            } else {
                std::snprintf(buf.data(), buf.size(), "%llu", static_cast<unsigned long long>(count));
            }
            return buf.data();
        };

        if (pipeline_stats.valid) {
            ImGui::Text("Triangles rendered (post-clip): %s (%llu)", fmt(pipeline_stats.clipped_primitive_count),
                        static_cast<unsigned long long>(pipeline_stats.clipped_primitive_count));

            if (pipeline_stats.mesh_stats_valid) {
                ImGui::Text("Task shader invocations: %s (%llu)", fmt(pipeline_stats.task_shader_invocation_count),
                            static_cast<unsigned long long>(pipeline_stats.task_shader_invocation_count));
                ImGui::Text("Mesh shader invocations: %s (%llu)", fmt(pipeline_stats.mesh_shader_invocation_count),
                            static_cast<unsigned long long>(pipeline_stats.mesh_shader_invocation_count));
            }
            ImGui::Text("Fragment shader invocations: %s (%llu)", fmt(pipeline_stats.fragment_shader_invocation_count),
                        static_cast<unsigned long long>(pipeline_stats.fragment_shader_invocation_count));
        } else {
            ImGui::TextDisabled("Pipeline stats not yet available");
        }

        ImGui::Text("Triangles submitted (pre-cull): %s (%u)", fmt(stats.submitted_triangle_count),
                    stats.submitted_triangle_count);
        ImGui::Text("Draw calls: %u  (opaque %u / mask %u / blend %u)", stats.indirect_command_count,
                    stats.opaque_indirect_count, stats.mask_indirect_count, stats.blend_indirect_count);
        ImGui::Text("Instances submitted: %s (%u)", fmt(stats.submitted_instance_count),
                    stats.submitted_instance_count);

        // Lags a frames-in-flight cycle behind the rest of the stats.
        auto const culled_percent = stats.submitted_instance_count != 0
                                            ? 100.0F * static_cast<float>(stats.visible_instance_count) /
                                                      static_cast<float>(stats.submitted_instance_count)
                                            : 0.0F;
        ImGui::Text("Instances visible (post-cull): %s (%u, %.1f%%)", fmt(stats.visible_instance_count),
                    stats.visible_instance_count, culled_percent);

        ImGui::Text("Model / mesh submissions: %u / %u", stats.model_submission_count, stats.mesh_submission_count);
        ImGui::Text("Lights: %u point / %u spot", stats.point_light_count, stats.spot_light_count);
    });

    widget("Frame timings", [&] {
        if (ImPlot::BeginPlot("Stage timings (cumulative ms)", ImVec2(-1, 250))) {
            ImPlot::SetupAxes("Frame", "ms", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisLimits(ImAxis_X1, timing_x - 600.0, timing_x, ImGuiCond_Always);

            constexpr auto first_stage = static_cast<std::uint32_t>(RenderStage::Culling);

            for (std::uint32_t stage = first_stage; stage < stage_count; ++stage) {
                auto const &buf = timing_buffers[stage];

                if (buf.data.empty()) {
                    continue;
                }

                ImPlotSpec spec;
                spec.Offset = buf.offset;
                spec.Stride = sizeof(ImVec2);
                spec.FillAlpha = 0.35F;

                std::string const label{to_string(static_cast<RenderStage>(stage))};

                if (stage == first_stage) {
                    ImPlot::PlotShaded(label.c_str(), &buf.data[0].x, &buf.data[0].y, static_cast<int>(buf.data.size()),
                                       0.0, spec);
                } else {
                    auto const &prev = timing_buffers[stage - 1];

                    ImPlotSpec prev_spec;
                    prev_spec.Offset = prev.offset;
                    prev_spec.Stride = sizeof(ImVec2);

                    ImPlot::PlotShaded(label.c_str(), &buf.data[0].x, &buf.data[0].y, &prev.data[0].y,
                                       static_cast<int>(buf.data.size()), prev_spec);
                }
            }

            ImPlot::EndPlot();
        }

        // Overlay time is already included in the forward/composite stages above.
        auto const &overlay_timings = renderer->last_frame_timings().overlays;

        if (!overlay_timings.empty() &&
            ImGui::BeginTable("Overlay timings", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Overlay");
            ImGui::TableSetupColumn("Stage");
            ImGui::TableSetupColumn("Prepare (ms)");
            ImGui::TableSetupColumn("Record (ms)");
            ImGui::TableHeadersRow();

            for (auto const &timing: overlay_timings) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(timing.name.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(timing.stage == OverlayStage::scene ? "scene" : "ui");
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", static_cast<double>(timing.prepare_milliseconds));
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", static_cast<double>(timing.record_milliseconds));
            }

            ImGui::EndTable();
        }
    });

    widget("Lighting", [&] {
        ImGui::SeparatorText("Debug");
        bool draw_light_icons = renderer->debug_draw_light_icons();
        if (ImGui::Checkbox("Draw light icons", &draw_light_icons)) {
            renderer->set_debug_draw_light_icons(draw_light_icons);
        }

        bool draw_physics_debug = debug_renderer->physics_debug_enabled();
        if (ImGui::Checkbox("Draw physics colliders", &draw_physics_debug)) {
            debug_renderer->set_physics_debug_enabled(draw_physics_debug);
        }

        bool draw_model_bounds_debug = debug_renderer->model_bounds_debug_enabled();
        if (ImGui::Checkbox("Draw model submesh bounds", &draw_model_bounds_debug)) {
            debug_renderer->set_model_bounds_debug_enabled(draw_model_bounds_debug);
        }

        bool meshlet_culling = renderer->meshlet_culling();
        if (ImGui::Checkbox("Meshlet culling (task shader)", &meshlet_culling)) {
            renderer->set_meshlet_culling(meshlet_culling);
        }

        auto light = renderer->directional_light();
        auto shadows = renderer->shadow_settings();
        bool dirty = false;

        dirty |= ImGui::SliderFloat("Azimuth", &light_azimuth_degrees, -180.0F, 180.0F, "%.1f deg");
        // Kept above the horizon; a horizontal light degenerates the cascade depth range.
        dirty |= ImGui::SliderFloat("Elevation", &light_elevation_degrees, 5.0F, 89.0F, "%.1f deg");
        dirty |= ImGui::ColorEdit3("Colour", &light.colour.x);
        dirty |= ImGui::SliderFloat("Intensity", &light.intensity, 0.0F, 10.0F);

        float ambient_intensity = renderer->ambient_intensity();
        if (ImGui::SliderFloat("Ambient intensity", &ambient_intensity, 0.0F, 1.0F)) {
            renderer->set_ambient_intensity(ambient_intensity);
        }

        ImGui::SeparatorText("Shadows");
        dirty |= ImGui::SliderFloat("Split lambda", &shadows.cascades.split_lambda, 0.0F, 1.0F);
        dirty |= ImGui::SliderFloat("Shadow distance", &shadows.cascades.shadow_distance, 20.0F, 500.0F);
        dirty |= ImGui::SliderFloat("PCF radius", &shadows.pcf_radius_texels, 0.5F, 4.0F);
        dirty |= ImGui::SliderFloat("Normal offset", &shadows.normal_offset_texels, 0.0F, 8.0F);
        dirty |= ImGui::SliderFloat("Depth bias", &shadows.depth_bias_world, 0.0F, 0.5F);
        dirty |= ImGui::SliderFloat("Bias slope", &shadows.depth_bias_slope, -8.0F, 0.0F);
        dirty |= ImGui::Checkbox("Cascade tint", &shadows.debug_cascade_tint);

        if (dirty) {
            auto const azimuth = glm::radians(light_azimuth_degrees);
            auto const elevation = glm::radians(light_elevation_degrees);

            light.direction = glm::normalize(glm::vec3{
                    std::cos(elevation) * std::cos(azimuth),
                    std::sin(elevation),
                    std::cos(elevation) * std::sin(azimuth),
            });

            renderer->set_directional_light(light);
            renderer->set_shadow_settings(shadows);
        }

        ImGui::SeparatorText("Fog");
        auto fog = renderer->fog_settings();
        bool fog_dirty = false;
        fog_dirty |= ImGui::Checkbox("Enabled", &fog.enabled);
        fog_dirty |= ImGui::ColorEdit3("Fog colour", &fog.colour.x);
        fog_dirty |= ImGui::SliderFloat("Fog extinction", &fog.extinction, 0.0F, 0.02F, "%.4f");
        fog_dirty |= ImGui::SliderFloat("Fog inscattering", &fog.inscattering, 0.0F, 2.0F);
        if (fog_dirty) {
            renderer->set_fog_settings(fog);
        }

        ImGui::SeparatorText("Punctual lights");
        auto &registry = active_scene()->get_registry();
        std::size_t index = 0;
        draw_rows(index, registry,
                  registry.view<Components::Transform, Components::PointLight, Components::GeneratedMeta>(),
                  draw_point_light);
        draw_rows(index, registry, registry.view<Components::Transform, Components::PointLight, Components::Meta>(),
                  draw_point_light);
        draw_rows(index, registry, registry.view<Components::Transform, Components::SpotLight, Components::Meta>(),
                  draw_spot_light);
        draw_rows(index, registry,
                  registry.view<Components::Transform, Components::SpotLight, Components::GeneratedMeta>(),
                  draw_spot_light);
    });


    if (auto const picked = model_browser.draw(editor_icons.get())) {
        switch (model_browse_target) {
            case ModelBrowseTarget::spawn_entity:
                spawn_streamed_model(*picked);
                break;

            case ModelBrowseTarget::inspector: {
                auto &registry = active_scene()->get_registry();
                if (registry.valid(model_browse_entity) && registry.all_of<Components::Model>(model_browse_entity)) {
                    // request() returns a reference for us, which set_entity_model() hands to the entity.
                    auto const model = renderer->model_streamer().request(*renderer, *picked, engine_models.cube,
                                                                          gui::path_to_utf8(picked->filename()));
                    set_entity_model(registry, model_browse_entity, model);
                }
                break;
            }
        }
        model_browse_entity = entt::null;
    }
}

auto Application::spawn_streamed_model(std::filesystem::path const &path) -> void {
    auto file_name = gui::path_to_utf8(path.filename());
    auto const model = renderer->model_streamer().request(*renderer, path, engine_models.cube, file_name);

    auto entity = Entity{active_scene(), gui::path_to_utf8(path.stem())};
    entity.emplace<Components::Transform>();
    entity.emplace<Components::Model>(Components::Model{.model = model});
    // Lets the entity's reference be released when it's removed or its model is swapped.
    entity.emplace<Components::StreamedModelTag>();
    selection_context().select(entity);

    model_loads.push_back(StreamedModelLoad{
            .scene = active_scene(),
            .entity = entity,
            .model = model,
            .file_name = std::move(file_name),
    });

    // Only settled entries are dropped, so a long-running load keeps reporting.
    while (model_loads.size() > max_listed_model_loads) {
        auto const settled = std::ranges::find_if(model_loads, &StreamedModelLoad::settled);
        if (settled == model_loads.end()) {
            break;
        }
        model_loads.erase(settled);
    }
}

auto Application::update_model_loads() -> void {
    auto &streamer = renderer->model_streamer();

    for (auto &load: model_loads) {
        if (load.settled) {
            continue;
        }

        auto const state = streamer.state(load.model);
        if (state == ModelRequestState::loading) {
            continue;
        }

        load.settled = true;

        if (state == ModelRequestState::failed) {
            load.status = std::format("Failed to load '{}': {}. Showing the placeholder cube.", load.file_name,
                                      streamer.failure_reason(load.model));
            continue;
        }

        if (load.model == engine_models.cube) {
            load.status = std::format("Could not reserve a model slot for '{}'. Showing the placeholder cube.",
                                      load.file_name);
            continue;
        }

        load.status = std::format("Loaded '{}'", load.file_name);

        // The collider is built from the real submesh bounds, which only exist once the model installed.
        if (load.scene != active_scene()) {
            continue;
        }
        auto &registry = load.scene->get_registry();
        if (!registry.valid(load.entity) || registry.all_of<Components::RigidBody>(load.entity)) {
            continue;
        }
        auto const *component = registry.try_get<Components::Model>(load.entity);
        if (component == nullptr || component->model != load.model) {
            continue;
        }
        if (auto const submesh_bounds = renderer->model_submesh_bounds(load.model)) {
            registry.emplace<Components::RigidBody>(load.entity,
                                                    Components::RigidBody::from_submesh_boxes(*submesh_bounds));
        }
    }
}

auto Application::set_entity_model(entt::registry &registry, entt::entity entity, ModelHandle model) -> void {
    auto const previous = registry.get<Components::Model>(entity).model;
    bool const owned_previous = registry.all_of<Components::StreamedModelTag>(entity);

    registry.patch<Components::Model>(entity, [&](Components::Model &component) { component.model = model; });
    registry.emplace_or_replace<Components::StreamedModelTag>(entity);

    // Released after the swap: if `model` == `previous`, the caller's reference replaces the entity's.
    if (owned_previous) {
        renderer->release_model(previous);
    }

    renderer->mark_dynamic_shadow_casters_dirty();
}

auto Application::play() -> void {
    // The runtime registry has its own entities.
    selection_context().clear();
    renaming_entity = entt::null;
    inspector_name_dirty = false;

    runtime_scene = std::make_unique<Scene>(*renderer);
    runtime_scene->physics_settings = editor_scene->physics_settings;
    game->clone_into_runtime(*editor_scene, *runtime_scene);

    // Set first so active_scene() resolves to runtime_scene below.
    is_playing = true;
    active_scene()->on_scene_start();
    active_scene()->attach_debug_renderer(*debug_renderer);

    // Each run gets a fresh PhysicsWorld, so terrain colliders from the previous run are stale.
    if (terrain) {
        terrain->on_physics_world_changed(active_scene()->physics_world.get());
    }

    game_mouse_captured = false;

    // Embedded play captures the cursor on the first Viewport click instead (see main.cxx).
    if (play_fullscreen) {
        capture_mouse();
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoKeyboard;
    }
}

auto Application::stop() -> void {
    selection_context().clear();
    renaming_entity = entt::null;
    inspector_name_dirty = false;

    // Before on_scene_stop() destroys the runtime PhysicsWorld.
    if (terrain) {
        terrain->on_physics_world_changed(nullptr);
    }

    active_scene()->detach_debug_renderer();
    debug_renderer->clear_lines();
    active_scene()->on_scene_stop();
    is_playing = false;
    game_mouse_captured = false;

    std::erase_if(model_loads, [&](StreamedModelLoad const &load) { return load.scene == runtime_scene.get(); });
    runtime_scene.reset();

    release_mouse();
    ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_NoKeyboard;
}

auto Application::capture_mouse() -> void {
    glfwSetInputMode(context.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    has_last_mouse_position = false;

    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouse;
}

auto Application::release_mouse() -> void {
    glfwSetInputMode(context.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    has_last_mouse_position = false;

    auto &io = ImGui::GetIO();
    io.ConfigFlags &= ~ImGuiConfigFlags_NoMouse;

    // GLFW moves the cursor back to where it was captured, but X11 reports no motion for that warp, so ImGui would
    // hover and click at the last virtual position until the mouse next moves.
    double x = 0.0;
    double y = 0.0;
    glfwGetCursorPos(context.window, &x, &y);
    io.AddMousePosEvent(static_cast<float>(x), static_cast<float>(y));
}

auto Application::update(float delta_time) -> void {
    ZoneScopedNC("ApplicationUpdate", tracy::Color::Firebrick);

    // Here rather than in the "Load Model" panel, which doesn't run while hidden or during fullscreen play.
    update_model_loads();


    // Keyed off the player's Transform rather than the follow camera, which springs and would jitter residency.
    if (terrain) {
        auto camera_xz = glm::vec2{camera.position().x, camera.position().z};

        if (is_playing) {
            auto &registry = active_scene()->get_registry();
            auto view = registry.view<Components::Transform const, Components::PlayerTag const>();

            for (auto &&[entity, transform]: view.each()) {
                camera_xz = glm::vec2{transform.position.x, transform.position.z};
                break;
            }
        }

        terrain->update(camera_xz);
    }

    std::erase_if(pending_deletions, [this](PendingDeletion &pending) {
        if (elapsed_time < pending.delete_at) {
            return false;
        }
        pending.commit();
        return true;
    });

    if (!is_playing) {
        return;
    }

    active_scene()->step(delta_time);

    game->on_update(*active_scene(), delta_time);
    systems::lifetime(active_scene()->get_registry(), *active_scene()->physics_world, delta_time);
}
auto Application::register_overlays() -> void {
    auto add = [this](OverlayDesc desc) {
        auto const name = desc.name;

        if (auto registration = renderer->register_overlay(std::move(desc)); registration) {
            overlays.push_back(std::move(*registration));
        } else {
            error("Could not register the '{}' overlay: {}", name, describe(registration.error()));
        }
    };

    add(OverlayDesc{
            .name = "Debug lines",
            .stage = OverlayStage::scene,
            .order = 0,
            .prepare = {},
            .record = [this](OverlayRecordContext const &overlay_context) { debug_renderer->record(overlay_context); },
    });

    add(OverlayDesc{
            .name = "ImGui",
            .stage = OverlayStage::ui,
            .order = 0,
            .prepare = {},
            .record =
                    [this](OverlayRecordContext const &overlay_context) {
                        imgui_renderer->render(overlay_context.command_buffer, overlay_context.frame_index);
                    },
    });
}

auto Application::on_startup() -> void {

    std::array const shader_directories{
            std::filesystem::path{"assets/shaders"},
    };
    if (!shader_watcher_.start(renderer->shader_change_queue(), shader_directories)) {
        error("Shader hot-reload watcher failed to start -- shaders will not live-reload this run");
    }
    imgui_renderer = std::make_unique<gui::ImGuiRenderer>(
            *renderer, gui::FontChoice{
                               .font_path = "assets/fonts/GoogleSansCode-Regular.ttf",
                               .size = 12,
                       });
    editor_icons = std::make_unique<gui::EditorIcons>(*renderer);
    register_overlays();
    renderer->queue_render_thread_event([this] {
        auto models = create_engine_models(*renderer);

        if (!models) {
            error("Fatal: could not create built-in engine models: {}", describe(models.error()));

            context.running.store(false, std::memory_order_release);
            glfwPostEmptyEvent();

            return;
        }

        engine_models = *models;

        game->on_populate(*editor_scene, *renderer, engine_models);

        if (auto terrain_info = game->terrain_create_info(*renderer)) {
            renderer->context().one_time_submit([this, info = *terrain_info](VkCommandBuffer command_buffer) {
                auto world = TerrainWorld::create(*renderer, command_buffer, info);

                if (!world) {
                    error("Fatal: could not create TerrainWorld: {}", world.error().message);

                    context.running.store(false, std::memory_order_release);
                    glfwPostEmptyEvent();

                    return;
                }

                terrain = std::make_unique<TerrainWorld>(std::move(*world));
            });
        }
    });
}

auto Application::on_event(KeyPressedEvent ev) -> bool {
    if (ev.key == GLFW_KEY_R && ev.modifiers == GLFW_MOD_CONTROL) {
        renderer->queue_render_thread_event([this] { game->on_populate(*editor_scene, *renderer, engine_models); });
    }
    if (ev.key == GLFW_KEY_F12) {
        renderer->request_screenshot();
    }

    if (is_playing) {
        // In embedded play the first Escape releases the mouse capture; the next one stops playing.
        if (ev.key == GLFW_KEY_ESCAPE) {
            if (game_mouse_captured) {
                game_mouse_captured = false;
                release_mouse();
                return true;
            }

            stop();
            return true;
        }

        game->on_key_pressed(*active_scene(), ev);
    } else {
        // 1-4 rather than W/E/R, which already move the editor camera.
        if (!ImGui::GetIO().WantCaptureKeyboard) {
            switch (ev.key) {
                case GLFW_KEY_1:
                    gizmo_operation = ImGuizmo::TRANSLATE;
                    break;
                case GLFW_KEY_2:
                    gizmo_operation = ImGuizmo::ROTATE;
                    break;
                case GLFW_KEY_3:
                    gizmo_operation = ImGuizmo::SCALE;
                    break;
                case GLFW_KEY_4:
                    gizmo_mode = gizmo_mode == ImGuizmo::WORLD ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
                    break;
                default:
                    break;
            }
        }

        camera.on_key_pressed(ev.key);
    }

    return true;
}
auto Application::on_event(KeyReleasedEvent ev) -> bool {
    if (is_playing) {
        game->on_key_released(*active_scene(), ev);
    } else {
        camera.on_key_released(ev.key);
    }

    return true;
}
auto Application::on_event(MouseMovedEvent ev) -> bool {
    if (is_playing) {
        // Embedded play only forwards look input while the Viewport has captured the mouse.
        if (play_fullscreen || game_mouse_captured) {
            game->on_mouse_moved(*active_scene(), ev);
        }
    } else {
        camera.on_mouse_moved(static_cast<float>(ev.delta_x), static_cast<float>(ev.delta_y), mouse_dragging);
    }

    return true;
}
auto Application::on_event(MouseScrolledEvent ev) -> bool {
    camera.on_mouse_scrolled(static_cast<float>(ev.delta_y));

    return true;
}
auto Application::on_event(MouseButtonPressedEvent ev) -> bool {
    if (ev.button == GLFW_MOUSE_BUTTON_RIGHT) {
        mouse_dragging = true;
    }

    if (is_playing) {
        game->on_mouse_button_pressed(*active_scene(), ev);
    }

    return true;
}
auto Application::on_event(MouseButtonReleasedEvent ev) -> bool {
    if (ev.button == GLFW_MOUSE_BUTTON_RIGHT) {
        mouse_dragging = false;
    }

    if (is_playing) {
        game->on_mouse_button_released(*active_scene(), ev);
    }

    return true;
}
