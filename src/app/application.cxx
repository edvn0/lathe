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
// DockBuilder* (default-layout construction, on_ui()'s dockspace host) lives
// here rather than in the public imgui.h.
#include "imgui_internal.h"
#include "rendering/debug_renderer.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/entity.hxx"
#include "rendering/imgui_renderer.hxx"
#include "rendering/imgui_widget.hxx"
#include "scene/components.hxx"
#include "scene/editor_camera.hxx"
#if MINGW_VULKAN_TRACK_MEMORY
#include "core/memory_tracking_ui.hxx"
#endif
#include "assets/shader_hot_reload_watcher.hxx"
#include "gpu/renderdoc.hxx"
#include "gpu/swapchain.hxx"
#include "physics/physics.hxx"
#include "physics/physics_world.hxx"
#include "portable-file-dialogs.h"
#include "rendering/renderer.hxx"
#include "rendering/renderer_application_policy.hxx"
#include "rendering/scene.hxx"

namespace {

    // Shared by the Hierarchy widget and the Inspector widget -- an entity's
    // display name is either its GeneratedMeta (runtime-spawned: bullets,
    // duplicates, loaded models) or its Meta (authored, interned FlyString),
    // falling back to "Entity <id>" when neither is present. GeneratedMeta is
    // treated as authoritative over Meta on the rare entity that has both
    // (Meta/GeneratedMeta are meant to be mutually exclusive -- see
    // Entity/GeneratedEntity in entity.hxx -- Scene's AttachedEntity, used
    // for IScript::on_attach/on_detach, no longer forces a stray empty Meta
    // onto an already-GeneratedMeta-named entity the way its predecessor
    // did). Kept as a defensive fallback rather than an assumed invariant,
    // so this and the Hierarchy widget's dedup stay in agreement about which
    // name "counts" if that ever recurs.
    [[nodiscard]] auto entity_display_name(entt::registry const &registry, entt::entity entity) -> std::string {
        char const *raw = nullptr;
        if (auto const *generated = registry.try_get<Components::GeneratedMeta>(entity)) {
            raw = generated->name.c_str();
        } else if (auto const *meta = registry.try_get<Components::Meta>(entity)) {
            raw = meta->name.c_str();
        }
        if (raw != nullptr && raw[0] != '\0') {
            return std::string(raw);
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

    // Rotation is stored as a quaternion (Components::Transform), but a
    // draggable-Euler field is what's actually usable in an inspector --
    // round-tripped through degrees on every edited frame rather than kept
    // as separate persistent Euler state, same tradeoff every editor built
    // on a quaternion transform makes (Unity included): fine for occasional
    // manual edits, not meant for continuous keyframed rotation.
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

        // Heightfield/compound shapes are baked from terrain/mesh bounds
        // (see PhysicsWorld::add_body and RigidBody::from_submesh_boxes) --
        // their shared_ptr payload has no sane hand-authored equivalent, so
        // they're shown but not switchable to/from here.
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

    // Copies every component in Cs... that `source` has onto `dest` -- backs
    // the Hierarchy widget's "Duplicate" context-menu action (see on_ui()).
    // Components::PhysicsBody is deliberately never passed in Cs...: it's a
    // non-owning handle into PhysicsWorld's Bullet arena (see its
    // declaration in physics_components.hxx), so copying it would leave two
    // entities pointing at the same live rigid body instead of each getting
    // their own once the scene is next played.
    template<typename... Cs>
    auto copy_components(entt::registry &registry, entt::entity source, entt::entity dest) -> void {
        (
                [&] {
                    // Empty tag types (PlayerTag, BulletTag) have no
                    // per-entity storage for entt's empty-type optimization
                    // to hand back a pointer to -- try_get<T>() on one of
                    // these fails to compile (registry.hpp tries to
                    // std::addressof() a void get()), so presence has to be
                    // tested and copied separately from stateful types.
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

#if !defined(_WIN32)
    // kdialog/zenity (spawned by pfd::open_file below) are themselves
    // dynamically linked against the system's Qt/GTK. pfd forks and
    // execvp()s them inheriting this process's environment, so if
    // LD_LIBRARY_PATH here points at a different Qt build (e.g. one
    // vendored alongside this engine), the child resolves its Qt libraries
    // from there instead, mismatches ABI, and crashes on startup -- which
    // looks like the dialog closing instantly with an empty selection
    // rather than a launch failure. Scope this around just the
    // pfd::open_file construction (the fork+exec happens synchronously
    // inside its constructor) to fix that without affecting anything else
    // this process spawns or dynamically loads.
    class ScopedLdLibraryPathClear {
    public:
        ScopedLdLibraryPathClear() : saved_(std::getenv("LD_LIBRARY_PATH") ? std::getenv("LD_LIBRARY_PATH") : "") {
            if (!saved_.empty()) {
                unsetenv("LD_LIBRARY_PATH");
            }
        }

        ~ScopedLdLibraryPathClear() {
            if (!saved_.empty()) {
                setenv("LD_LIBRARY_PATH", saved_.c_str(), 1);
            }
        }

        ScopedLdLibraryPathClear(ScopedLdLibraryPathClear const &) = delete;
        auto operator=(ScopedLdLibraryPathClear const &) -> ScopedLdLibraryPathClear & = delete;

    private:
        std::string saved_;
    };
#endif

    // Shared by the "Load Model" widget and the Inspector's per-entity
    // "Change Model" browse control -- both open the same glTF/GLB picker.
    auto open_model_dialog(std::unique_ptr<pfd::open_file> &dialog) -> void {
        // Logs the exact helper command (zenity/kdialog/osascript/...) pfd
        // resolves to on stderr, useful if it ever silently falls back to
        // "echo" because no supported helper was found.
        pfd::settings::verbose(true);

#if !defined(_WIN32)
        ScopedLdLibraryPathClear const scoped_ld_library_path_clear;
#endif
        dialog = std::make_unique<pfd::open_file>(
                "Load model", ".", std::vector<std::string>{"glTF models", "*.gltf *.glb", "All files", "*"});
    }

    // Call only once `dialog->ready(0)` is true. Resets `dialog` either way;
    // returns nullopt if the picker was cancelled (empty selection).
    [[nodiscard]] auto consume_model_dialog(std::unique_ptr<pfd::open_file> &dialog)
            -> std::optional<std::filesystem::path> {
        auto const selection = dialog->result();
        dialog.reset();

        if (selection.empty()) {
            return std::nullopt;
        }

        return std::filesystem::path{selection.front()};
    }

    // Moved to include/rendering/imgui_widget.hxx so game code (IGame::on_ui())
    // can use the same helper -- kept unqualified here via this using-declaration
    // so none of the call sites below needed to change.
    using gui::widget;

    // Shared between the Assets panel's Materials section (where entries get
    // queued into Application::pending_deletions -- see its doc comment in
    // application.hxx) and the Inspector's Material Override picker/menu,
    // so a material about to actually be deleted a few seconds from now
    // can't also be newly attached to an entity in the meantime.
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
    // Fullscreen play covers the whole swapchain with no editor chrome at
    // all, exactly like this engine's play mode always has -- see
    // Renderer::record_frame's matching `fullscreen` branch, which this must
    // stay in lockstep with: it decides whether the 3D scene lands straight
    // in the swapchain or in the offscreen viewport_target the Viewport
    // panel below displays.
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

    // Builds the default layout exactly once ever -- imgui.ini (see
    // ImGuiRenderer::set_app_name) restores this dockspace's node before
    // this code runs on every launch after the first, so rebuilding it
    // unconditionally would silently discard the user's layout every time.
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

        // Drawn straight into the Viewport window's own draw list -- rather
        // than a separate overlay window sized to match its rect (the old
        // approach, back when the 3D view was a fullscreen backdrop and not
        // an actual ImGui window) -- so it's always on top of the Image()
        // above regardless of window z-order; two same-rect windows aren't
        // reliably ordered relative to each other. ImGuizmo does its own
        // mouse hit-testing against io.MousePos rather than relying on the
        // host window being hovered, so this works fine even though the
        // Viewport window itself takes normal input (camera-look drag,
        // click-to-capture in embedded play).
        if (!is_playing && has_room) {
            auto &registry = active_scene()->get_registry();

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

                auto matrix = registry.get<Components::Transform>(selected_entity).matrix();

                if (ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(projection), gizmo_operation, gizmo_mode,
                                         glm::value_ptr(matrix))) {
                    auto const translation = glm::vec3{matrix[3]};
                    glm::vec3 const scale{glm::length(glm::vec3{matrix[0]}), glm::length(glm::vec3{matrix[1]}),
                                          glm::length(glm::vec3{matrix[2]})};
                    glm::mat3 const rotation_matrix{glm::vec3{matrix[0]} / scale.x, glm::vec3{matrix[1]} / scale.y,
                                                    glm::vec3{matrix[2]} / scale.z};
                    auto const rotation = glm::quat_cast(rotation_matrix);

                    // patch<>() rather than a direct write so
                    // Scene::on_transform_changed still fires (e.g. to
                    // re-dirty light data when gizmo-editing a light).
                    registry.patch<Components::Transform>(selected_entity, [&](Components::Transform &transform) {
                        transform.position = translation;
                        transform.rotation = rotation;
                        transform.scale = scale;
                    });

                    // Interactive drag, not a discrete edit -- bounded-staleness
                    // refresh via the dynamic flag rather than forcing every
                    // cascade to redraw on every dragged frame.
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

        if (!model_load_dialog) {
            if (ImGui::Button("Browse...")) {
                open_model_dialog(model_load_dialog);
            }
        } else {
            ImGui::BeginDisabled();
            ImGui::Button("Browse...");
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextDisabled("Waiting for file dialog...");

            if (!model_load_dialog->ready(0))
                return;

            auto const selected_path = consume_model_dialog(model_load_dialog);
            if (!selected_path)
                return;

            auto const &path = *selected_path;
            auto const model =
                    renderer->model_streamer().request(*renderer, path, engine_models.cube, path.filename().string());

            if (model.valid()) {
                auto entity = Entity{active_scene(), path.stem().string()};
                entity.emplace<Components::Transform>();
                entity.emplace<Components::Model>(Components::Model{.model = model});
                // Enables Renderer::destroy_model() on Hierarchy "remove" --
                // see StreamedModelTag's doc comment.
                entity.emplace<Components::StreamedModelTag>();
                auto const submesh_bounds = renderer->model_submesh_bounds(model);
                if (submesh_bounds) {
                    entity.emplace<Components::RigidBody>(Components::RigidBody::from_submesh_boxes(*submesh_bounds));
                }
                // model_streamer().request() returns an already-installed
                // handle immediately for a path it previously finished
                // loading (see ModelStreamer::path_cache_) -- that shows up
                // here as submesh_bounds already being available, since a
                // still-loading model reports its fallback's bounds.
                model_load_status = submesh_bounds ? std::format("Reused already-loaded '{}'", path.filename().string())
                                                   : std::format("Loading '{}'...", path.filename().string());
            } else {
                model_load_status =
                        std::format("Failed to load '{}': could not reserve a model slot", path.filename().string());
            }
        }

        if (!model_load_status.empty()) {
            ImGui::TextUnformatted(model_load_status.c_str());
        }
    });

    widget("Assets", [&] {
        auto &assets = renderer->assets();

        // Recursively lists files under `root` whose (lowercased)
        // extension is in `extensions`. Best-effort discovery for this
        // panel, not a build step -- a missing/unreadable directory just
        // yields no results instead of throwing.
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

        // Default entry renderer used by every draw_file_backed_section
        // caller except Textures, which passes draw_texture_entry below
        // instead to show a live thumbnail alongside the name.
        auto const draw_bullet_entry = [](auto const &entry) {
            ImGui::BulletText("%s%s", entry.name.c_str(), entry.handle.valid() ? "" : " (invalid)");
        };

        // Models/Textures: registered entries, plus on-disk files under
        // `root` not yet loaded (one-click "Load" funnels through the same
        // request-based path the Inspector's "Browse..." uses, so a freshly
        // loaded file is immediately selectable everywhere else).
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
                    continue; // already loaded, listed above
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

        // Textures get a small live thumbnail next to the name instead of
        // draw_bullet_entry's plain text -- ImageHandle's .index doubles as
        // its ImTextureID (see EditorIcons::texture()'s identical cast), so
        // this reads straight from the GPU-resident texture the same way
        // the viewport does, with no separate CPU-side preview to keep in
        // sync.
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

        // Scripts aren't a file asset -- no disk scan, just whatever's
        // currently registered (see AssetRegistry's doc comment). Unlike
        // Materials below, there's no in-panel way to author a new one
        // (scripts are C++ types registered by game code via
        // ScriptStorage::register_script), so this stays a read-only list.
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

        // Every editable field of a material, shared between the "New
        // Material" creation popup and each registered material's inline
        // editor below -- edits `info` in place and reports whether
        // anything changed, same draw_fields(T&) -> bool convention the
        // Inspector's section() helper uses for component fields.
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

            // One combo per texture slot, offering every registered texture
            // plus a "(default)" entry that falls back to that slot's
            // engine-wide fallback image (ImageStorage::white()/
            // flat_normal()/...) -- the same fallback load_model.cxx uses
            // for a glTF material that didn't specify that slot.
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
                    continue; // Queued for deletion -- shown in "Recently deleted" below instead.
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
                        // Not destroyed yet -- see PendingDeletion's doc
                        // comment. The entry stays fully live in
                        // AssetRegistry/MaterialStorage (and thus still
                        // usable by any entity's MaterialOverride) until the
                        // grace period actually elapses.
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

            // Anything queued above shows here instead, with a live
            // countdown and a Restore button that cancels the pending
            // commit -- cheap to offer since nothing was actually mutated
            // yet, so "restoring" is just forgetting the queue entry.
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

                // At most one of these fires per frame -- both are indices
                // into the same vector, and restoring/committing anything
                // but the last one picked would invalidate the other.
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

        // Toggle-button toolbar (move/rotate/scale + a local/world space
        // switch) rather than radio buttons/a checkbox -- an icon strip
        // reads at a glance the way Unreal's and Godot's viewport gizmo
        // toolbars do, and keeps the 1-4 shortcuts visually anchored to
        // the buttons they drive.
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

        // Case-insensitive substring filter typed into the search box below.
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

        // Entities carry no explicit "type" -- this mirrors the views below
        // by branching on the same ECS components, just picking an icon
        // instead of a view filter.
        auto const visual_for = [&](entt::entity entity) -> EntityVisual {
            if (registry.all_of<Components::PointLight>(entity)) {
                return {gui::EditorIcon::point_light, ImVec4(1.00F, 0.84F, 0.35F, 1.0F)};
            }
            if (registry.all_of<Components::SpotLight>(entity)) {
                return {gui::EditorIcon::spot_light, ImVec4(1.00F, 0.84F, 0.35F, 1.0F)};
            }
            if (registry.all_of<Components::PlayerTag>(entity)) {
                return {gui::EditorIcon::player, ImVec4(0.47F, 0.86F, 0.55F, 1.0F)};
            }
            if (registry.all_of<Components::BulletTag>(entity)) {
                return {gui::EditorIcon::bullet, ImVec4(1.00F, 0.53F, 0.38F, 1.0F)};
            }
            if (registry.all_of<Components::Model>(entity) || registry.all_of<Components::InstancedModel>(entity)) {
                return {gui::EditorIcon::mesh, ImVec4(0.88F, 0.64F, 0.37F, 1.0F)};
            }
            if (registry.all_of<Components::Script>(entity)) {
                return {gui::EditorIcon::script, ImVec4(0.42F, 0.70F, 1.00F, 1.0F)};
            }
            return {gui::EditorIcon::empty, ImVec4(0.60F, 0.60F, 0.64F, 1.0F)};
        };

        // Bullets grouped under a collapsible node instead of listed flat --
        // shoot_bullet() can spawn a dozen at once per Ctrl+click, each with
        // only a ~3s Lifetime, which would otherwise dominate/spam this list.
        auto const bullet_view =
                registry.view<Components::Transform, Components::GeneratedMeta, Components::BulletTag>();
        auto const bullet_count = static_cast<std::uint32_t>(std::distance(bullet_view.begin(), bullet_view.end()));

        // Two separate views (rather than one over both Meta and
        // GeneratedMeta) because an entity only ever carries one of the two
        // name-component types in the common case -- see entity_display_name
        // above for the defensive fallback if that's ever not true, which
        // listed_entities' dedup below relies on to avoid listing -- and
        // PushID'ing -- such an entity twice. Both exclude BulletTag since
        // those are listed above instead.
        auto const meta_view =
                registry.view<Components::Transform, Components::Meta>(entt::exclude<Components::BulletTag>);
        auto const generated_view =
                registry.view<Components::Transform, Components::GeneratedMeta>(entt::exclude<Components::BulletTag>);

        // generated_view first, then meta_view entries not already added --
        // an entity with both components (see entity_display_name's comment
        // above) must land in listed_entities exactly once, or it gets
        // PushID'd -- and drawn -- twice at the same tree level, which is a
        // genuine ImGui duplicate-ID collision between the two rows, not
        // just a cosmetic one.
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

        // Every listed entity above is bucketed as either a root or a child
        // of another listed entity via Components::Parent -- e.g. each
        // house's 7 wall/roof/chimney parts hang off one "house_N" entity
        // and each tree's trunk+canopy hang off one "tree_N" entity (see
        // BasicGame::on_populate), so the panel shows them nested instead of
        // as one large flat list.

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

        // A parent row stays visible while a search filter is active as
        // long as it or any of its descendants match -- otherwise typing
        // e.g. "chimney" would hide the house_N row and, with it, its own
        // matching chimney child.
        std::unordered_map<entt::entity, bool> match_cache;
        std::function<bool(entt::entity)> subtree_matches = [&](entt::entity entity) -> bool {
            if (auto const cached = match_cache.find(entity); cached != match_cache.end()) {
                return cached->second;
            }
            // Seed with `true` before recursing so a malformed/cyclic Parent
            // chain can't cause infinite recursion.
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

        // Set from inside draw_entity_node's per-row context menu (and the
        // background one below the tree) as the user clicks a menu item;
        // applied once after every row has been drawn rather than
        // immediately, so a Delete/Duplicate never mutates the registry
        // out from under the view/tree walk that's still iterating it this
        // frame.
        enum class HierarchyAction : std::uint8_t { none, add_child, duplicate, remove, reparent };
        HierarchyAction pending_action = HierarchyAction::none;
        // add_child: the new entity's parent (entt::null = root-level).
        // duplicate/remove: the entity the action applies to.
        // reparent: the new parent (entt::null = drop-to-root), paired with
        // drag_reparent_source below for the entity being moved.
        entt::entity action_target = entt::null;
        // Set by a drag-and-drop drop (see draw_entity_node and the
        // root-level drop zone below) to the entity that was dragged.
        entt::entity drag_reparent_source = entt::null;

        std::size_t row_index = 0;

        // Draws one row (icon + label), plus -- if `entity` has any listed
        // children -- an expandable tree node nesting them recursively.
        // Leaf and parent rows share the same icon/selection/zebra-stripe
        // chrome; only the underlying clickable widget differs (Selectable
        // vs TreeNodeEx), since a plain Selectable can't expand/collapse.
        std::function<void(entt::entity)> draw_entity_node = [&](entt::entity entity) {
            if (!subtree_matches(entity)) {
                return;
            }

            auto const name = entity_display_name(registry, entity);
            auto const child_it = children_of.find(entity);
            bool const has_children = child_it != children_of.end() && !child_it->second.empty();

            auto visual = visual_for(entity);
            if (has_children && visual.icon == gui::EditorIcon::empty) {
                // A pure grouping entity (house_N/tree_N) with no type of
                // its own -- flag it as a container the same way the
                // Bullets group is, rather than the generic "empty" icon.
                visual = {gui::EditorIcon::folder, ImVec4(0.95F, 0.80F, 0.45F, 1.0F)};
            }

            ImGui::PushID(static_cast<int>(entity));

            float const row_height = ImGui::GetFrameHeight();
            float const icon_size = ImGui::GetTextLineHeight();
            ImVec2 const row_pos = ImGui::GetCursorPos();
            ImVec2 const row_screen_pos = ImGui::GetCursorScreenPos();
            float const avail_width = ImGui::GetContentRegionAvail().x;

            // Subtle zebra striping (same TableRowBgAlt token the rest of
            // the theme reserves for it) so a dense list stays scannable
            // without a real ImGui table.
            if (row_index++ % 2 == 1) {
                ImGui::GetWindowDrawList()->AddRectFilled(
                        row_screen_pos, ImVec2(row_screen_pos.x + avail_width, row_screen_pos.y + row_height),
                        ImGui::GetColorU32(ImGuiCol_TableRowBgAlt));
            }

            bool const is_selected = selected_entity == entity;
            float label_x = row_pos.x + style.FramePadding.x;
            ImVec2 next_row_pos;
            bool open = false;

            if (has_children) {
                ImGuiTreeNodeFlags const flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnArrow |
                                                 ImGuiTreeNodeFlags_FramePadding |
                                                 (is_selected ? ImGuiTreeNodeFlags_Selected : ImGuiTreeNodeFlags_None);
                // TreeNodeEx sizes its row from its own (visible) label text
                // height -- with the fully-empty label "##node" used to have
                // (nothing before the "##" survives past the ID), that
                // collapsed to a row shorter than row_height/the leaf
                // Selectable rows below, throwing off spacing between
                // parent and leaf rows. A single leading space " " gives it
                // real (if invisible) text to measure, and
                // ImGuiTreeNodeFlags_FramePadding makes it use the same
                // FramePadding-based height formula GetFrameHeight() does --
                // together these make its natural height exactly
                // row_height, matching every other row in this tree.
                open = ImGui::TreeNodeEx(" ##node", flags);
                // OpenOnArrow keeps the arrow the only thing that
                // expands/collapses -- clicking the rest of the row just
                // selects it, matching Unreal/Godot outliner behaviour.
                if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                    selected_entity = entity;
                }
                next_row_pos = ImGui::GetCursorPos();
                label_x = row_pos.x + ImGui::GetTreeNodeToLabelSpacing();
            } else {
                if (ImGui::Selectable("##row", is_selected, ImGuiSelectableFlags_None,
                                      ImVec2(avail_width, row_height))) {
                    selected_entity = entity;
                }
                next_row_pos = ImGui::GetCursorPos();
            }

            // Targets whichever of TreeNodeEx/Selectable above just ran --
            // both are the full-width "last item" for this row, so a
            // right-click anywhere across it (not just over the icon/label
            // overlay drawn below) opens the menu. OpenPopupOnItemClick
            // returns void in the imgui version this project is pinned to
            // (CMakeLists.txt's v1.92.5-docking put back the pre-1.77
            // signature), so selection-on-right-click is handled separately
            // via IsItemClicked rather than that call's return value.
            if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                selected_entity = entity;
            }
            ImGui::OpenPopupOnItemClick("entity_context", ImGuiPopupFlags_MouseButtonRight);

            // Drag-and-drop reparenting: dragging a row onto another row
            // makes the dragged entity a child of the row it's dropped on,
            // matching Unreal/Godot outliner behaviour. entt::entity is a
            // trivially-copyable scoped enum, so the raw value is stashed
            // directly in the payload rather than round-tripped through
            // entt::to_integral.
            if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
                ImGui::SetDragDropPayload("HIERARCHY_ENTITY", &entity, sizeof(entity));
                ImGui::TextUnformatted(name.c_str());
                ImGui::EndDragDropSource();
            }

            if (ImGui::BeginDragDropTarget()) {
                if (auto const *payload = ImGui::AcceptDragDropPayload("HIERARCHY_ENTITY")) {
                    entt::entity dropped{};
                    std::memcpy(&dropped, payload->Data, sizeof(dropped));
                    // Deferred like every other mutation here -- applied
                    // once the whole tree has finished drawing (see
                    // pending_action's declaration above), since
                    // reparenting mid-walk would shuffle children_of out
                    // from under the recursion that's still consuming it
                    // this frame.
                    pending_action = HierarchyAction::reparent;
                    action_target = entity;
                    drag_reparent_source = dropped;
                }
                ImGui::EndDragDropTarget();
            }

            // The clickable widget above already consumed this row's layout
            // slot; icon + label are drawn as a manually-positioned overlay
            // on top of it, then the cursor is restored to right after it so
            // the next row/sibling isn't thrown off by wherever these
            // overlay draws happen to leave it.
            ImGui::SetCursorPos(ImVec2(label_x, row_pos.y + (row_height - icon_size) * 0.5F));
            ImGui::ImageWithBg(editor_icons->texture(visual.icon), ImVec2(icon_size, icon_size), ImVec2(0, 0),
                               ImVec2(1, 1), ImVec4(0, 0, 0, 0), visual.tint);

            ImGui::SetCursorPos(ImVec2(label_x + icon_size + style.ItemInnerSpacing.x,
                                       row_pos.y + (row_height - ImGui::GetTextLineHeight()) * 0.5F));
            ImGui::TextUnformatted(name.c_str());

            ImGui::SetCursorPos(next_row_pos);

            if (has_children && open) {
                for (auto const child: child_it->second) {
                    draw_entity_node(child);
                }
                ImGui::TreePop();
            }

            if (ImGui::BeginPopup("entity_context")) {
                if (ImGui::MenuItem("Add Child Entity")) {
                    pending_action = HierarchyAction::add_child;
                    action_target = entity;
                }
                if (ImGui::MenuItem("Duplicate")) {
                    pending_action = HierarchyAction::duplicate;
                    action_target = entity;
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Delete")) {
                    pending_action = HierarchyAction::remove;
                    action_target = entity;
                }
                ImGui::EndPopup();
            }

            ImGui::PopID();
        };

        auto const total_count = bullet_count + static_cast<std::uint32_t>(listed_entities.size());
        ImGui::TextDisabled("%u %s", total_count, total_count == 1 ? "entity" : "entities");

        bool const any_bullet_matches =
                search_lower.empty() || std::ranges::any_of(bullet_view, [&](entt::entity e) {
                    return matches_filter(registry.get<Components::GeneratedMeta>(e).name.c_str());
                });

        if (bullet_count > 0 && any_bullet_matches) {
            ImGui::PushID("bullets_group");

            float const row_height = ImGui::GetFrameHeight();
            float const icon_size = ImGui::GetTextLineHeight();
            ImVec2 const row_pos = ImGui::GetCursorPos();

            // See draw_entity_node's comment on the equivalent TreeNodeEx
            // call for why the label is " ##bullets_node" rather than
            // "##bullets_node" and FramePadding is added -- both are needed
            // for this row to naturally come out row_height tall, matching
            // every other row in the tree instead of the shorter height an
            // empty-label unframed tree node collapses to.
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

        // The rows above finish by restoring the cursor with SetCursorPos()
        // rather than a real item (see draw_entity_row) -- if the last one
        // drawn happens to also be the last thing in the window, ImGui's
        // ErrorCheckUsingSetCursorPosToExtendParentBoundaries() asserts
        // unless a real item follows to confirm the window's content
        // bounds. A zero-size Dummy() satisfies that cheaply.
        ImGui::Dummy(ImVec2(0.0F, 0.0F));

        // Dropping a dragged row onto the empty space below the tree
        // clears its parent, moving it back to root level.
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

        // Right-click on empty space below/between rows -- NoOpenOverItems
        // keeps this from also firing over a row (each row already has its
        // own "entity_context" popup via OpenPopupOnItemClick above).
        if (ImGui::BeginPopupContextWindow("hierarchy_bg_context",
                                           ImGuiPopupFlags_MouseButtonRight | ImGuiPopupFlags_NoOpenOverItems)) {
            if (ImGui::MenuItem("Create Empty Entity")) {
                pending_action = HierarchyAction::add_child;
                action_target = entt::null;
            }
            ImGui::EndPopup();
        }

        // Deferred from the context menus above (see pending_action's
        // declaration) -- applied now that every row for this frame has
        // finished drawing.
        switch (pending_action) {
            case HierarchyAction::add_child: {
                auto new_entity = GeneratedEntity{active_scene(), "Entity"};
                // Every listed/drawn entity is required to carry a Transform
                // (generated_view/meta_view above both filter on it) --
                // without this the new entity would be created but never
                // show up in the tree.
                new_entity.emplace<Components::Transform>();
                if (action_target != entt::null && registry.valid(action_target)) {
                    new_entity.emplace<Components::Parent>(Components::Parent{.entity = action_target});
                }
                selected_entity = new_entity;
                break;
            }
            case HierarchyAction::duplicate: {
                if (registry.valid(action_target)) {
                    // Recreates `source`'s subtree under `parent` (entt::null
                    // for root-level), reusing children_of so a duplicated
                    // house_N/tree_N brings its parts along instead of
                    // leaving them behind on the original.
                    std::function<entt::entity(entt::entity, entt::entity)> duplicate_subtree =
                            [&](entt::entity source, entt::entity parent) -> entt::entity {
                        auto const clone = registry.create();

                        copy_components<Components::Transform, Components::Model, Components::InstancedModel,
                                        Components::RigidBody, Components::MaterialOverride, Components::PlayerTag,
                                        Components::Lifetime, Components::PointLight, Components::SpotLight,
                                        Components::Script, Components::BulletTag>(registry, source, clone);

                        // Duplicates are always dynamically named like other
                        // runtime-created entities (bullets etc.) via
                        // GeneratedMeta, regardless of whether the source
                        // used Meta (FlyString) or GeneratedMeta -- see
                        // entity_display_name's comment above on why the two
                        // aren't mixed on one entity.
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

                    auto const *source_parent = registry.try_get<Components::Parent>(action_target);
                    selected_entity = duplicate_subtree(action_target,
                                                        source_parent != nullptr ? source_parent->entity : entt::null);
                }
                break;
            }
            case HierarchyAction::remove: {
                if (registry.valid(action_target)) {
                    std::function<void(entt::entity)> delete_subtree = [&](entt::entity target) {
                        if (auto const it = children_of.find(target); it != children_of.end()) {
                            for (auto const child: it->second) {
                                delete_subtree(child);
                            }
                        }
                        if (selected_entity == target) {
                            selected_entity = entt::null;
                        }

                        // Only entities the "Load Model" widget tagged have a
                        // ref-counted ModelHandle safe to destroy_model() on
                        // removal -- see StreamedModelTag's doc comment.
                        if (auto const *model = registry.try_get<Components::Model>(target);
                            model != nullptr && registry.all_of<Components::StreamedModelTag>(target)) {
                            static_cast<void>(renderer->destroy_model(model->model));
                        }

                        registry.destroy(target);
                    };
                    delete_subtree(action_target);
                }
                break;
            }
            case HierarchyAction::reparent: {
                if (registry.valid(drag_reparent_source) && drag_reparent_source != action_target) {
                    // Walk the prospective new parent's own ancestor chain
                    // looking for the dragged entity -- dropping an entity
                    // onto one of its own descendants would otherwise
                    // create a cycle that get_world_transform()'s
                    // depth-capped walk (scene.cxx) would then silently cut
                    // off rather than reject outright.
                    bool creates_cycle = false;
                    for (auto walk = action_target; walk != entt::null && registry.valid(walk);) {
                        if (walk == drag_reparent_source) {
                            creates_cycle = true;
                            break;
                        }
                        auto const *walk_parent = registry.try_get<Components::Parent>(walk);
                        walk = walk_parent != nullptr ? walk_parent->entity : entt::null;
                    }

                    if (!creates_cycle) {
                        if (action_target == entt::null) {
                            registry.remove<Components::Parent>(drag_reparent_source);
                        } else {
                            registry.emplace_or_replace<Components::Parent>(
                                    drag_reparent_source, Components::Parent{.entity = action_target});
                        }
                    }
                }
                break;
            }
            case HierarchyAction::none:
                break;
        }

        if (selected_entity != entt::null && !registry.valid(selected_entity)) {
            selected_entity = entt::null;
        }
    });

    widget("Inspector", [&] {
        auto &registry = active_scene()->get_registry();

        if (selected_entity == entt::null || !registry.valid(selected_entity)) {
            ImGui::TextDisabled("No entity selected");
            return;
        }

        ImGui::TextUnformatted(entity_display_name(registry, selected_entity).c_str());
        ImGui::Separator();

        // One collapsible section per present component. `draw_fields`
        // edits the component in place and returns whether it changed
        // (drives patch<T>, which is what fires Scene::on_transform_changed/
        // mark_lights_dirty -- see connect_light_signals in scene.cxx); a
        // false-returning draw_fields is a read-only/marker section instead.
        // The header's own close button ("x") queues removal rather than
        // removing mid-CollapsingHeader, matching the Hierarchy widget's
        // pending-action-applied-after-the-fact pattern above.
        auto section = [&]<typename T>(char const *label, auto &&draw_fields) {
            if (!registry.all_of<T>(selected_entity)) {
                return;
            }

            bool open = true;
            ImGui::PushID(label);
            if (ImGui::CollapsingHeader(label, &open, ImGuiTreeNodeFlags_DefaultOpen)) {
                if (draw_fields(registry.get<T>(selected_entity))) {
                    registry.patch<T>(selected_entity);

                    // Interactive field drag, not a discrete edit -- see the
                    // matching call in the ImGuizmo block above. Model/
                    // MaterialOverride add-or-remove (below) needs no such
                    // call: current_shadow_scene_signature is recomputed
                    // from the actual caster batches every frame, so a
                    // structural change there is already caught next frame
                    // without per-entity tracking.
                    if constexpr (std::is_same_v<T, Components::Transform>) {
                        renderer->mark_dynamic_shadow_casters_dirty();
                    }
                }
            }
            ImGui::PopID();

            if (!open) {
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

            bool changed = false;
            auto &models = renderer->assets().models();
            auto const current_name = models.name_of(model.model);

            // Reassigns model.model to `new_handle`, releasing the old
            // handle first if this entity owns a ref-counted reference to
            // it (see StreamedModelTag's doc comment) -- safe to call
            // repeatedly (destroy_model()/GeometryArena both actually
            // reclaim now, see AssetRegistry's module comment).
            auto const reassign = [&](ModelHandle new_handle) {
                if (new_handle == model.model || !new_handle.valid()) {
                    return;
                }
                if (registry.all_of<Components::StreamedModelTag>(selected_entity)) {
                    static_cast<void>(renderer->destroy_model(model.model));
                }
                model.model = new_handle;
                registry.emplace_or_replace<Components::StreamedModelTag>(selected_entity);
                changed = true;
            };

            if (ImGui::BeginCombo("Asset", current_name.empty() ? "(unnamed)" : std::string(current_name).c_str())) {
                for (auto const &entry: models.entries()) {
                    bool const is_selected = entry.handle == model.model;
                    if (ImGui::Selectable(entry.name.c_str(), is_selected)) {
                        // A combo-picked handle is a second owner of an
                        // already-registered (and thus already-owned)
                        // handle -- retain it first, matching how
                        // Renderer::load_model's cache-hit path does the
                        // same before handing an existing handle to a new
                        // caller.
                        renderer->retain_model(entry.handle);
                        reassign(entry.handle);
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::SameLine();
            if (!inspector_model_dialog) {
                if (ImGui::Button("Browse...##model")) {
                    open_model_dialog(inspector_model_dialog);
                }
            } else {
                ImGui::BeginDisabled();
                ImGui::Button("Browse...##model");
                ImGui::EndDisabled();
                ImGui::TextDisabled("Waiting for file dialog...");

                if (inspector_model_dialog->ready(0)) {
                    if (auto const path = consume_model_dialog(inspector_model_dialog)) {
                        // request() itself handles ref-counting for both a
                        // fresh load (ref_count starts at 1) and a
                        // path-cache hit (retains internally) -- no extra
                        // retain_model() needed here, unlike the combo path
                        // above.
                        reassign(renderer->model_streamer().request(*renderer, *path, engine_models.cube,
                                                                    path->filename().string()));
                    }
                }
            }

            return changed;
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
                    // Not offered as a new pick once queued for deletion
                    // (see PendingDeletion's doc comment) -- still shown
                    // above as the current selection if it's already set,
                    // since it remains genuinely valid until the grace
                    // period actually elapses.
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
        // PlayerTag/BulletTag are empty types -- entt's empty-type storage
        // optimization means registry.get<T>() on one returns void (nothing
        // is actually stored per-entity, only presence), so they can't go
        // through `section` above, which assumes get<T>() yields a T& to
        // hand to draw_fields.
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

            // Everything else here is positional (lights render at the
            // entity's Transform, RigidBody syncs it every physics step --
            // see PhysicsWorld::step), so Transform has to exist first.
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
                    // Bullet body construction takes effect on next physics
                    // population (e.g. re-entering Play), not live -- see
                    // Scene::on_scene_start/PhysicsWorld::populate_from.
                    registry.emplace<Components::RigidBody>(selected_entity);
                }

                // Positional like the above, so gated on has_transform too.
                // Picked by name via AssetRegistry rather than a hand-typed
                // index/generation pair -- see the Model section above for
                // why that used to be unsafe.
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
        // Editable any time, including mid-play, so a running embedded
        // session can flip to fullscreen (or back) without stopping.
        ImGui::Checkbox("Fullscreen", &play_fullscreen);
    });

    widget("Scene stats", [&] {
        auto const &stats = renderer->last_frame_stats();
        auto const &pipeline_stats = renderer->last_frame_pipeline_stats();

        // Helper lambda to get a formatted string in-line
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

        // Lags submitted_instance_count by one frames-in-flight cycle -- see
        // RendererFrame::culled_readback_buffer's comment -- so it can
        // briefly read 0 or a slightly stale count right after a scene
        // change, not just "nothing survived culling".
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

                if (stage == first_stage) {
                    ImPlot::PlotShaded(to_string(static_cast<RenderStage>(stage)).data(), &buf.data[0].x,
                                       &buf.data[0].y, static_cast<int>(buf.data.size()), 0.0, spec);
                } else {
                    auto const &prev = timing_buffers[stage - 1];

                    ImPlotSpec prev_spec;
                    prev_spec.Offset = prev.offset;
                    prev_spec.Stride = sizeof(ImVec2);

                    ImPlot::PlotShaded(to_string(static_cast<RenderStage>(stage)).data(), &buf.data[0].x,
                                       &buf.data[0].y, &prev.data[0].y, static_cast<int>(buf.data.size()), prev_spec);
                }
            }

            ImPlot::EndPlot();
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
        // Clamped so the light never goes horizontal -- that
        // degenerates the ortho depth range in shadow_cascades.cxx.
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
}

auto Application::play() -> void {
    // Names an entity in editor_scene's registry; runtime_scene (about to be
    // created below) starts with entirely different entity handles.
    selected_entity = entt::null;

    runtime_scene = std::make_unique<Scene>(*renderer);
    runtime_scene->physics_settings = editor_scene->physics_settings;
    game->clone_into_runtime(*editor_scene, *runtime_scene);

    // is_playing flips first so active_scene() already resolves to
    // runtime_scene for on_scene_start()/attach_debug_renderer() below.
    is_playing = true;
    active_scene()->on_scene_start();
    active_scene()->attach_debug_renderer(*debug_renderer);

    // PhysicsWorld is recreated wholesale per run (on_scene_start() above
    // just made a fresh one) -- every terrain collider handle TerrainWorld
    // held from a previous run belonged to a PhysicsWorld that's already
    // gone. GPU slots are untouched; colliders rebind lazily under
    // TerrainWorld's normal per-frame budget.
    if (terrain) {
        terrain->on_physics_world_changed(active_scene()->physics_world.get());
    }

    game_mouse_captured = false;

    // Fullscreen play covers the whole window like this engine's play mode
    // always has -- capture the cursor immediately. Embedded play leaves the
    // editor (and its cursor) alone until the user clicks into the Viewport
    // panel -- see mouse_button_callback in main.cxx -- so Hierarchy/
    // Inspector/etc. stay usable around the running game.
    if (play_fullscreen) {
        glfwSetInputMode(context.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        has_last_mouse_position = false;

        // While the cursor is disabled its reported position is an unbounded
        // virtual accumulator, not a real screen coordinate -- ImGui would
        // otherwise keep hit-testing widgets against wherever that value
        // drifts to (starting at wherever the "Play" click happened to land)
        // and intermittently steal input meant for the game.
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouse | ImGuiConfigFlags_NoKeyboard;
    }
}

auto Application::stop() -> void {
    // Names an entity in runtime_scene's registry, which is reset() below.
    selected_entity = entt::null;

    // Before on_scene_stop() tears down the runtime PhysicsWorld, so
    // TerrainWorld never holds a handle into an instance that's already
    // been destroyed.
    if (terrain) {
        terrain->on_physics_world_changed(nullptr);
    }

    active_scene()->detach_debug_renderer();
    debug_renderer->clear_lines();
    active_scene()->on_scene_stop();
    // Flips active_scene() over to editor_scene immediately -- nothing below
    // this line touches the runtime scene, so runtime_scene.reset() is safe.
    is_playing = false;
    game_mouse_captured = false;
    runtime_scene.reset();

    glfwSetInputMode(context.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    ImGui::GetIO().ConfigFlags &= ~(ImGuiConfigFlags_NoMouse | ImGuiConfigFlags_NoKeyboard);
}

auto Application::update(float delta_time) -> void {
    ZoneScopedNC("ApplicationUpdate", tracy::Color::Firebrick);

    // Residency selection only (no GPU work -- see TerrainWorld::update) --
    // runs regardless of is_playing, so terrain streams in the editor too.
    // Keyed off the player entity's own Transform rather than the follow
    // camera: PlayerCamera springs and gets pulled by wall occlusion, which
    // would otherwise jitter which chunks are considered resident.
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

    // Runs regardless of is_playing, like terrain streaming above -- a
    // material deleted from the Assets panel while in the editor should
    // still actually get reclaimed a few seconds later even if the user
    // never presses Play. See PendingDeletion's doc comment (application.hxx)
    // for why deletions go through this queue instead of firing immediately.
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
            // create_engine_models() above already went through this same
            // one-time-submit path internally (via create_model_from_cpu_data),
            // so nesting another one here -- still on the render thread,
            // still serialized through queue_render_thread_event -- is safe.
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
        // Escape is the keyboard way out of play mode -- previously the
        // only way back to the editor was tabbing/alt-tabbing out and
        // relying on focus_callback's stop() (see main.cxx), which is not
        // a deliberate exit path. Consumed here rather than forwarded to
        // the game, since play mode is ending.
        //
        // Embedded play captured the mouse on a Viewport-panel click
        // (mouse_button_callback, main.cxx) rather than unconditionally on
        // play() the way fullscreen does -- so the first Escape there just
        // releases that capture back to the editor (leaving the rest of the
        // editor, and the running game, untouched); a second Escape (mouse
        // no longer captured) falls through to actually stopping, same as
        // fullscreen play always has.
        if (ev.key == GLFW_KEY_ESCAPE) {
            if (game_mouse_captured) {
                game_mouse_captured = false;
                glfwSetInputMode(context.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                return true;
            }

            stop();
            return true;
        }

        game->on_key_pressed(*active_scene(), ev);
    } else {
        // Gated on WantCaptureKeyboard so typing an entity name (were that
        // ever added to the Hierarchy widget) or similar text entry doesn't
        // also swap the gizmo operation. 1-4 rather than the ImGuizmo-classic
        // W/E/R: those already fly the editor camera (EditorCamera's
        // is_forward_key/is_up_key etc.), and camera.on_key_pressed below is
        // unconditional, so reusing them would move the camera and the
        // gizmo mode at once.
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
        // Fullscreen play captures the cursor unconditionally in play(), so
        // every delta while playing is deliberate look input, same as
        // always. Embedded play leaves the cursor free to roam the rest of
        // the editor until a Viewport-panel click captures it (see
        // mouse_button_callback, main.cxx) -- without this gate, moving the
        // mouse over Hierarchy/Inspector/etc. while an embedded session runs
        // would spuriously spin the game's camera.
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
