#include <fstream>
#include <charconv>
#include <csignal>
#include <map>
#include <memory>
#include <print>
#include <random>
#include <ranges>
#include <volk.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <mutex>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <expected>
#include <filesystem>
#include <fstream>
#include <format>
#include <future>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <entt/entt.hpp>

#include "rendering/file_browser.hxx"
#include "runtime/engine_main.hxx"
#include "runtime/engine_cvars.hxx"
#include "runtime/runtime.hxx"
#include "assets/texture_pipeline.hxx"
#include "core/cvar.hxx"
#include "core/game_manifest.hxx"
#include "core/project_file.hxx"
#include "core/paths.hxx"
#include "core/resources.hxx"
#include "serialisation/resource_pack.hxx"
#include "runtime/fatal_dialog.hxx"
#include "runtime/frame_clock.hxx"
#include "game/game.hxx"
#include "assets/file_change_watcher.hxx"
#include "assets/shader_bake.hxx"
#include "assets/gpu_texture_encoder.hxx"
#include "assets/gpu_texture_service.hxx"
#include "assets/shader_pack.hxx"
#include "core/human_readable_bytes.hxx"
#include "assets/shader_pack_cache.hxx"
#include "storage/caches.hxx"
#include "core/allocator.hxx"
#include "core/config.hxx"
#include "core/command_line.hxx"
#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/thread_pool.hxx"
#include "core/random.hxx"
#include "glm/gtc/type_ptr.hpp"
#include "gpu/context.hxx"
#include "gpu/device_wait.hxx"
#include "gpu/queue_selection.hxx"
#include "gpu/swapchain.hxx"
#include "imgui.h"
#include "implot.h"
#include "maths/aabb.hxx"
#include "physics/physics.hxx"
#include "physics/physics_world.hxx"
#include "rendering/cluster_grid.hxx"
#include "rendering/debug_renderer.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/entity.hxx"
#include "rendering/imgui_renderer.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "rendering/screenshot.hxx"
#include "scene/components.hxx"
#include "scene/selection_context.hxx"
#include "scene/editor_camera.hxx"
#include "runtime/vulkan_bootstrap.hxx"

namespace {

    template<typename T>
    concept HasDepth = requires(T t) {
        { t.depth };
    };

    template<typename T>
    concept HasHeight = requires(T t) {
        { t.height };
    };

    template<typename T>
    concept HasWidth = requires(T t) {
        { t.width };
    };

    constexpr auto compare = []<typename A, typename B>(A const &a, B const &b) -> bool {
        if constexpr (HasDepth<A> && HasDepth<B>) {
            return a.width == b.width && a.height == b.height && a.depth == b.depth;
        } else if constexpr ((HasHeight<A> && !HasDepth<A>) && (HasHeight<B> && !HasDepth<B>) ) {
            return a.width == b.width && a.height == b.height;
        } else if constexpr ((HasWidth<A> && !HasHeight<A>) && (HasWidth<B> && !HasHeight<B>) ) {
            return a.width == b.width;
        } else {
            return false;
        }
    };

    // The same colours the forward shader gives a meshlet in the debug colour view (pcg hash of the meshlet index).
    [[nodiscard]] auto meshlet_debug_colour(std::uint32_t cluster_id) -> glm::vec3 {
        auto const state = (cluster_id * 747796405U) + 2891336453U;
        auto const word = ((state >> ((state >> 28U) + 4U)) ^ state) * 277803737U;
        auto const hash = (word >> 22U) ^ word;

        auto const hue = static_cast<float>(hash & 0xFFFFU) / 65535.0F;
        auto const value = 0.75F + (0.25F * (static_cast<float>((hash >> 16U) & 0xFFU) / 255.0F));
        constexpr float saturation = 0.65F;

        auto const channel = [&](float offset) {
            auto const fractional = hue + offset - std::floor(hue + offset);
            return std::clamp(std::abs((fractional * 6.0F) - 3.0F) - 1.0F, 0.0F, 1.0F);
        };

        return value * glm::mix(glm::vec3{1.0F}, glm::vec3{channel(0.0F), channel(2.0F / 3.0F), channel(1.0F / 3.0F)},
                                saturation);
    }

    // Each cone: apex at the meshlet centre (pushed out by extrude like the shader does), opening along the average
    // normal. meshopt's cutoff is the sine of the cone's half angle; a cutoff of 1 means no usable cone.
    auto draw_meshlet_cones(debug_draw::DebugRenderer &lines, std::vector<MeshletCone> const &cones,
                            glm::mat4 const &world, float extrude, float shrink) -> void {
        constexpr int rim_segments = 12;
        constexpr float max_cutoff = 0.985F;
        constexpr float two_pi = 6.2831853F;

        for (std::size_t index = 0; index < cones.size(); ++index) {
            auto const &cone = cones[index];

            if (cone.cutoff >= max_cutoff) {
                continue;
            }

            auto const colour = meshlet_debug_colour(static_cast<std::uint32_t>(index));
            auto const apex = cone.centre + (cone.axis * (extrude * cone.radius));
            auto const length = cone.radius * (1.0F - shrink) * 0.9F;
            auto const half_angle = std::asin(cone.cutoff);
            auto const rim_radius = length * std::sin(half_angle);

            auto const helper = std::abs(cone.axis.y) < 0.9F ? glm::vec3{0.0F, 1.0F, 0.0F} : glm::vec3{1.0F, 0.0F, 0.0F};
            auto const side = glm::normalize(glm::cross(cone.axis, helper));
            auto const up = glm::cross(cone.axis, side);
            auto const base = apex + (cone.axis * (length * std::cos(half_angle)));

            auto const to_world = [&](glm::vec3 const &point) { return glm::vec3{world * glm::vec4{point, 1.0F}}; };

            auto previous = base + (side * rim_radius);
            for (int step = 1; step <= rim_segments; ++step) {
                auto const angle = two_pi * static_cast<float>(step) / static_cast<float>(rim_segments);
                auto const next = base + (((side * std::cos(angle)) + (up * std::sin(angle))) * rim_radius);
                lines.add_line(to_world(previous), to_world(next), colour);
                if (step % 3 == 0) {
                    lines.add_line(to_world(apex), to_world(next), colour);
                }
                previous = next;
            }
            lines.add_line(to_world(apex), to_world(base), colour);
        }
    }

    auto submit_scene(Runtime &runtime) -> std::expected<void, RendererError> {
        ZoneScopedNC("SubmitScene", tracy::Color::RoyalBlue);

        auto &registry = runtime.active_scene()->get_registry();

        runtime.renderer->set_environment(runtime.active_scene()->environment);

        auto view = registry.view<Components::Transform const, Components::Model const>();

        // Entities under shared parents share the walk up them (see systems::WorldTransformCache).
        static thread_local auto world_transforms = systems::WorldTransformCache{};
        world_transforms.begin_frame();

        auto const draw_model_bounds = runtime.debug_renderer->model_bounds_debug_enabled();
        constexpr auto model_bounds_debug_colour = glm::vec3{0.2F, 1.0F, 0.4F};

        // Editor selection is drawn with the same outline games get from Components::Outlined. The selection is read
        // once: asking it per entity locks it every time.
        auto selected = std::vector<entt::entity>{};
        auto submesh = SelectionContext::SubmeshHighlight{};
        if (!runtime.is_playing) {
            auto snapshot = selection_context().snapshot();
            selected = std::move(snapshot.entities);
            submesh = snapshot.submesh;
            std::ranges::sort(selected);
        }
        // A singled-out material slot is outlined instead of its whole entity.
        auto const is_selected = [&selected, &submesh](entt::entity entity) {
            return !selected.empty() && std::ranges::binary_search(selected, entity) &&
                   !(submesh.active() && submesh.entity == entity);
        };
        auto const outline_source_of = [&submesh](entt::entity entity) {
            return submesh.active() && submesh.entity == entity ? submesh.material : MaterialHandle{};
        };

        // Looked up once for all entities, not once per entity by type. Through a const registry, a component nobody
        // has used gets no storage made (a null pointer here).
        auto const &read_only = std::as_const(registry);
        auto const *const outlines = read_only.storage<Components::Outlined>();
        auto const *const overrides = read_only.storage<Components::MaterialOverride>();
        auto const *const parents = read_only.storage<Components::Parent>();
        auto const *const cones_storage = read_only.storage<Components::DebugMeshletCones>();

        // The debug drawing below is not safe to run on several threads (and is rare): with it on, the loop is serial.
        auto const debug_draw = draw_model_bounds || runtime.debug_renderer->meshlet_cones_debug_enabled() ||
                                (cones_storage != nullptr && !cones_storage->empty());

        // Static models are drawn from the baked batches below, and left out of the loop that follows. The debug
        // drawing needs every model's world transform, so with it on they are submitted one by one like the rest.
        auto &static_models = runtime.static_models;
        if (debug_draw) {
            static_models.clear();
        } else {
            static_models.update(registry, runtime.active_scene()->static_revision(), selected,
                                 [&renderer = *runtime.renderer](ModelHandle model) { return renderer.has_model(model); });
        }

        if (!debug_draw) {
            // Entities are looked at in the pool's order, as view.each() does, so what is submitted does not change.
            // The renderer's submissions are opened, filled straight from several threads (each entity is independent
            // of the others) and closed. Slot overrides, which are appended to one shared list, are left for after. A
            // thread keeps its own cache of parent chains (see WorldTransformCache).
            struct Fixup {
                std::size_t index = 0;
                entt::entity entity = entt::null;
            };

            auto const &pool = *view.handle();

            // With static models baked, the entities left to submit are listed once, so the renderer is asked for room
            // for those and not for the whole pool.
            auto const filtered = static_models.entity_count() != 0;
            auto dynamic_entities = std::vector<entt::entity>{};
            if (filtered) {
                dynamic_entities.reserve(pool.size() - std::min(pool.size(), static_models.entity_count()));
                for (auto const entity: pool) {
                    if (!static_models.contains(entity)) {
                        dynamic_entities.push_back(entity);
                    }
                }
            }
            auto const total = filtered ? dynamic_entities.size() : pool.size();

            auto const submissions = runtime.renderer->begin_model_submissions(total);
            auto const count = submissions.size();
            if (count < total) {
                error("Could not submit all scene objects: the renderer holds {} models", count);
            }

            auto fixups = std::vector<Fixup>{};
            auto fixups_mutex = std::mutex{};
            auto skipped = std::atomic<std::size_t>{0};
            auto outlined_count = std::atomic<std::size_t>{0};
            auto const renderer = runtime.renderer.get();

            auto const fill = [&](std::size_t first, std::size_t last) {
                auto local_fixups = std::vector<Fixup>{};
                auto local_skipped = std::size_t{0};
                auto local_outlined = std::size_t{0};
                auto local_transforms = systems::WorldTransformCache{};

                auto entities = pool.begin() + static_cast<std::ptrdiff_t>(first);
                for (auto index = first; index < last; ++index, ++entities) {
                    auto const entity = filtered ? dynamic_entities[index] : *entities;
                    auto &submission = submissions[index];

                    if (!view.contains(entity)) {
                        submission.skip = true;
                        ++local_skipped;
                        continue;
                    }

                    auto const &transform = view.get<Components::Transform const>(entity);
                    auto const &model = view.get<Components::Model const>(entity);

                    if (!renderer->has_model(model.model)) {
                        submission.skip = true;
                        ++local_skipped;
                        continue;
                    }

                    auto const parented = parents != nullptr && parents->contains(entity);
                    auto const has_slots = overrides != nullptr && overrides->contains(entity) &&
                                           !overrides->get(entity).slots.empty();

                    submission.model = model.model;
                    submission.transform = parented ? local_transforms.get_with_parent(registry, entity, transform)
                                                    : transform.matrix();
                    submission.material_override =
                            overrides != nullptr && overrides->contains(entity) ? overrides->get(entity).material
                                                                                 : MaterialHandle{};
                    submission.slot_override_first = 0;
                    submission.slot_override_count = 0;
                    submission.outlined = (outlines != nullptr && outlines->contains(entity)) || is_selected(entity);
                    submission.outline_source = submission.outlined ? MaterialHandle{} : outline_source_of(entity);
                    submission.skip = false;
                    local_outlined += (submission.outlined || submission.outline_source.valid()) ? 1U : 0U;

                    if (has_slots) {
                        local_fixups.push_back(Fixup{.index = index, .entity = entity});
                    }
                }

                skipped += local_skipped;
                outlined_count += local_outlined;
                if (!local_fixups.empty()) {
                    auto const lock = std::scoped_lock{fixups_mutex};
                    fixups.insert(fixups.end(), local_fixups.begin(), local_fixups.end());
                }
            };

            constexpr auto parallel_threshold = std::size_t{4096};
            if (count < parallel_threshold) {
                fill(0, count);
            } else {
                thread_pool().submit_blocks(std::size_t{0}, count, fill).wait();
            }

            // Blocks finish in any order; the overrides list is filled in the pool's order.
            std::ranges::sort(fixups, {}, &Fixup::index);
            for (auto const &fixup: fixups) {
                runtime.renderer->set_model_submission_slot_overrides(
                        submissions[fixup.index],
                        std::span<MaterialSlotOverride const>{overrides->get(fixup.entity).slots});
            }

            if (auto result = runtime.renderer->end_model_submissions(skipped.load(), outlined_count.load()); !result) {
                error("Some scene objects have a model that does not exist: {}", describe(result.error()));
            }

            for (auto const &batch: static_models.batches()) {
                auto result = runtime.renderer->submit_model_instances(batch.model, batch.transforms,
                                                                       batch.material_override, batch.revision);
                if (!result) {
                    error("Could not submit static models (model index {}, {} instances): {}", batch.model.index,
                          batch.transforms.size(), describe(result.error()));
                }
            }
        } else {
            for (auto [entity, transform, model]: view.each()) {
                auto const outlined = registry.all_of<Components::Outlined>(entity) || is_selected(entity);
                auto const outline_source = outlined ? MaterialHandle{} : outline_source_of(entity);
                auto const *override_component = registry.try_get<Components::MaterialOverride const>(entity);
                auto const material_override =
                        override_component != nullptr ? override_component->material : MaterialHandle{};
                auto const slot_overrides = override_component != nullptr
                                                    ? std::span<MaterialSlotOverride const>{override_component->slots}
                                                    : std::span<MaterialSlotOverride const>{};

                auto const world_transform = world_transforms.get(registry, entity, transform);
                auto result = runtime.renderer->submit_model(model.model, world_transform, material_override,
                                                             slot_overrides, outlined, outline_source);

                if (!result) {
                    error("Could not submit scene object (model index {}): {}", model.model.index,
                          describe(result.error()));
                }

                auto const *cones = registry.try_get<Components::DebugMeshletCones const>(entity);
                if (cones != nullptr && cones->enabled) {
                    draw_meshlet_cones(*runtime.debug_renderer, runtime.renderer->model_meshlet_cones(model.model),
                                       world_transform, cones->extrude, cones->shrink);
                } else if (runtime.debug_renderer->meshlet_cones_debug_enabled()) {
                    draw_meshlet_cones(*runtime.debug_renderer, runtime.renderer->model_meshlet_cones(model.model),
                                       world_transform, 0.0F, 0.0F);
                }

                if (draw_model_bounds) {
                    auto const submesh_bounds = runtime.renderer->model_submesh_bounds(model.model);

                    if (submesh_bounds) {
                        for (auto const &[min, max]: *submesh_bounds) {
                            auto const [world_min, world_max] = maths::transform_aabb(world_transform, min, max);
                            runtime.debug_renderer->add_aabb(world_min, world_max, model_bounds_debug_colour);
                        }
                    }
                }
            }
        }

        auto instanced_model_view = registry.view<Components::InstancedModel const>();

        for (auto [entity, instanced]: instanced_model_view.each()) {
            auto result = runtime.renderer->submit_model_instances(
                    instanced.model, instanced.transforms, instanced.material_override, instanced.revision,
                    instanced.palette_offsets);

            if (!result) {
                error("Could not submit instanced model (model index {}, {} instances): {}", instanced.model.index,
                      instanced.transforms.size(), describe(result.error()));
            }
        }

        auto point_light_view = registry.view<Components::Transform const, Components::PointLight const>();

        for (auto [entity, transform, light]: point_light_view.each()) {
            auto const world_transform = systems::get_world_transform(registry, entity, transform);

            auto result = runtime.renderer->submit_point_light(Renderer::PointLight{
                    .position = glm::vec3{world_transform[3]},
                    .colour = light.colour,
                    .intensity = light.intensity,
                    .range = light.range,
                    .id = std::uint64_t{entt::to_integral(entity)} + 1,
                    .casts_shadows = light.casts_shadows,
            });

            if (!result) {
                error("Could not submit point light: {}", describe(result.error()));
            }
        }

        auto decal_view = registry.view<Components::Transform const, Components::Decal const>();

        for (auto [entity, transform, decal]: decal_view.each()) {
            if (!decal.material.valid()) {
                continue;
            }

            auto result = runtime.renderer->submit_decal(Renderer::Decal{
                    .transform = systems::get_world_transform(registry, entity, transform),
                    .material = decal.material,
                    .opacity = decal.opacity,
                    .angle_fade_start_degrees = decal.angle_fade_start_degrees,
                    .angle_fade_end_degrees = decal.angle_fade_end_degrees,
                    .sort_order = decal.sort_order,
            });

            if (!result) {
                error("Could not submit decal: {}", describe(result.error()));
            }
        }

        auto spot_light_view = registry.view<Components::Transform const, Components::SpotLight const>();

        for (auto [entity, transform, light]: spot_light_view.each()) {
            auto const world_transform = systems::get_world_transform(registry, entity, transform);
            auto const direction = glm::normalize(glm::mat3{world_transform} * glm::vec3{0.0F, -1.0F, 0.0F});

            auto result = runtime.renderer->submit_spot_light(Renderer::SpotLight{
                    .position = glm::vec3{world_transform[3]},
                    .direction = direction,
                    .colour = light.colour,
                    .intensity = light.intensity,
                    .range = light.range,
                    .inner_cone_degrees = light.inner_cone_degrees,
                    .outer_cone_degrees = light.outer_cone_degrees,
                    .id = std::uint64_t{entt::to_integral(entity)} + 1,
                    .casts_shadows = light.casts_shadows,
            });

            if (!result) {
                error("Could not submit spot light: {}", describe(result.error()));
            }
        }

        if (auto *const terrain = runtime.active_terrain(); terrain != nullptr) {
            terrain->submit(*runtime.renderer);
        }

        // Water bodies (Components::Water) and planets (Components::Planet), placed from this frame's camera.
        runtime.water_system.submit(*runtime.renderer, registry, glm::vec3{glm::inverse(runtime.frame_view)[3]});
        runtime.planet_system.submit(*runtime.renderer, registry,
                                     PlanetView{.origin = runtime.frame_origin,
                                                .camera = glm::dvec3{glm::inverse(runtime.frame_view)[3]},
                                                .time = runtime.elapsed_time});

        return {};
    }

    auto initialize_runtime(VulkanContext &context, Runtime &runtime) noexcept -> bool {
        auto renderer_result = runtime.renderer->initialize(RendererCreateInfo{
                .extent = context.swapchain.extent(),
                .geometry_capacity = 256UZ * 1024UZ * 1024UZ,
                .material_capacity = 4096,
                .mesh_capacity = 8192,
                .model_capacity = 4096,
                .pipeline_capacity = 1024,
                .swapchain_format = context.swapchain.format(),
                .hdr_output = context.swapchain.hdr_active(),
                .paper_white_nits = context.hdr_paper_white_nits,
                .peak_nits = context.hdr_peak_nits,
                .samples = VK_SAMPLE_COUNT_4_BIT,
        });

        if (!renderer_result) {
            error("Could not initialize renderer: {}", describe(renderer_result.error()));
            return false;
        }

        return true;
    }

    auto to_begin_error(QueueSetError queue_error) noexcept -> SwapchainBeginFrameError {
        return SwapchainBeginFrameError{
                .kind = queue_error.kind == QueueSetError::Kind::device_lost
                                ? SwapchainBeginFrameError::Kind::device_lost
                                : SwapchainBeginFrameError::Kind::fatal_error,
                .context = std::move(queue_error.context),
        };
    }

    auto begin_gpu_frame(VulkanContext &context,
                         FrameClock &clock) noexcept -> std::expected<SwapchainFrame, SwapchainBeginFrameError> {
        auto const slot = context.swapchain.current_slot();

        {
            auto const waiting = clock.scope(CpuPhase::slot_wait);
            if (auto begun = context.queue_set.begin_slot(slot); !begun) {
                return std::unexpected(to_begin_error(std::move(begun.error())));
            }
        }

        auto frame = [&] {
            auto const acquiring = clock.scope(CpuPhase::acquire);
            return context.swapchain.acquire(slot);
        }();
        if (!frame) {
            return std::unexpected(std::move(frame.error()));
        }

        auto buffer = context.queue_set.command_buffer(frame_graph::LogicalQueue::graphics);
        if (!buffer) {
            return std::unexpected(to_begin_error(std::move(buffer.error())));
        }

        frame->command_buffer = *buffer;

        return *frame;
    }

    auto end_gpu_frame(VulkanContext &context, SwapchainFrame const &frame, std::span<SubmitBatch const> planned,
                       FrameClock &clock) noexcept -> SwapchainFrameResult {
        auto submitting = std::optional<FrameClock::Scope>{};
        submitting.emplace(clock, CpuPhase::submit);

        if (auto const ended = vkEndCommandBuffer(frame.command_buffer); ended != VK_SUCCESS) {
            error("vkEndCommandBuffer failed with VkResult {}", static_cast<int>(ended));

            return ended == VK_ERROR_DEVICE_LOST ? SwapchainFrameResult::device_lost
                                                 : SwapchainFrameResult::fatal_error;
        }

        std::vector<SubmitBatch> batches{planned.begin(), planned.end()};
        if (batches.empty()) {
            batches.push_back(SubmitBatch{
                    .queue = frame_graph::LogicalQueue::graphics,
                    .command_buffer = frame.command_buffer,
                    .signal_index = 0,
                    .waits_swapchain_acquire = true,
                    .signals_render_finished = true,
            });
        }

        auto const submitted = context.queue_set.submit(batches, context.swapchain.image_available(frame.frame_index),
                                                        context.swapchain.render_finished(frame.image_index));
        if (!submitted) {
            return submitted.error().kind == QueueSetError::Kind::device_lost ? SwapchainFrameResult::device_lost
                                                                              : SwapchainFrameResult::fatal_error;
        }

        submitting.reset();
        auto const presenting = clock.scope(CpuPhase::present);
        return context.swapchain.present(frame);
    }

    auto draw(VulkanContext &context, Runtime &runtime, FrameClock &clock) noexcept -> bool {
        ZoneScopedNC("Draw", tracy::Color::RoyalBlue);

        auto frame = begin_gpu_frame(context, clock);

        if (!frame) {
            switch (frame.error().kind) {
                case SwapchainBeginFrameError::Kind::recreated:
                    return true;

                case SwapchainBeginFrameError::Kind::device_lost:
                    error("The Vulkan device was lost");
                    context.device_lost.store(true, std::memory_order_release);

                    return false;

                case SwapchainBeginFrameError::Kind::fatal_error:
                    if (frame.error().context.has_value()) {
                        error("Could not begin swapchain frame: {}", describe(*frame.error().context));
                    } else {
                        error("Could not begin swapchain frame");
                    }

                    return false;
            }

            return false;
        }

        // The camera comes first: the scene submission places camera-relative content (the floating origin, the ocean
        // grid) from it.
        auto const active_aspect = runtime.renderer->aspect(frame->frame_index);
        auto const active_camera = runtime.is_playing ? runtime.game->camera(*runtime.active_scene(), active_aspect)
                                                      : runtime.editing_camera(active_aspect);

        runtime.frame_view = active_camera.view;
        runtime.frame_projection = active_camera.projection;
        runtime.frame_origin = active_camera.origin;

        auto frame_ok = true;
        {
            auto const submitting = clock.scope(CpuPhase::scene_submit);

            if (auto *const terrain = runtime.active_terrain(); terrain != nullptr) {
                terrain->process_ready(*runtime.renderer, frame->command_buffer,
                                       runtime.active_scene()->physics_world.get());
            }

            auto submit_result = submit_scene(runtime);
            if (!submit_result) {
                error("Could not submit scene: {}", describe(submit_result.error()));
                frame_ok = false;
            }
        }

        if (frame_ok) {
            {
                auto const ui = clock.scope(CpuPhase::ui);
                runtime.imgui_renderer->begin_frame(frame->extent, frame->frame_index);
                runtime.on_ui(frame->frame_index);
                runtime.imgui_renderer->end_frame();
            }

            auto const preparing = clock.scope(CpuPhase::prepare);
            auto prepare_result = runtime.renderer->prepare_frame(
                    frame->command_buffer,
                    {
                            .view = active_camera.view,
                            .projection = active_camera.projection,
                            .near_clip = active_camera.near_clip,
                            .far_clip = active_camera.far_clip,
                            .vertical_fov_radians = active_camera.vertical_fov_radians,
                            .aspect_ratio = active_aspect,
                            .time = runtime.elapsed_time,
                    },
                    frame->frame_index);

            if (!prepare_result) {
                error("Could not prepare renderer frame: {}", describe(prepare_result.error()));

                frame_ok = false;
            }
        }

        if (frame_ok) {
            auto const recording = clock.scope(CpuPhase::record);
            auto record_result = runtime.renderer->record_frame(FrameRecordInfo{
                    .command_buffer = frame->command_buffer,
                    .swapchain_image =
                            SwapchainImage{
                                    .image = frame->image,
                                    .view = frame->image_view,
                                    .format = frame->format,
                                    .extent = frame->extent,
                            },
                    .frame_index = frame->frame_index,
                    .composite_target = runtime.is_playing && runtime.play_fullscreen
                                                ? CompositeTarget::swapchain
                                                : CompositeTarget::viewport_panel,
            });

            if (!record_result) {
                error("Could not record renderer frame: {}", describe(record_result.error()));

                frame_ok = false;
            }
        }

        auto const end_result = end_gpu_frame(context, *frame, runtime.renderer->submit_batches(), clock);
        clock.mark_present();

        if (!frame_ok) {
            return false;
        }

        switch (end_result) {
            case SwapchainFrameResult::success:
            case SwapchainFrameResult::recreated:
                return true;

            case SwapchainFrameResult::device_lost:
                error("The Vulkan device was lost");
                context.device_lost.store(true, std::memory_order_release);

                return false;

            case SwapchainFrameResult::fatal_error:
                error("Could not submit or present frame");

                return false;
        }

        return false;
    }

    auto request_resize_if_needed(VulkanContext &context, int width, int height) noexcept -> void {
        if (!context.framebuffer_dirty.exchange(false, std::memory_order_acq_rel)) {
            return;
        }

        context.swapchain.request_recreate(VkExtent2D{
                .width = static_cast<std::uint32_t>(width),
                .height = static_cast<std::uint32_t>(height),
        });
    }

    struct WindowData {
        VulkanContext *ctx;
        Runtime *app;
    };

    auto framebuffer_size_callback(GLFWwindow *window, int width, int height) noexcept -> void {
        auto *context = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->ctx;

        if (context == nullptr) {
            return;
        }

        context->framebuffer_width.store(width, std::memory_order_relaxed);
        context->framebuffer_height.store(height, std::memory_order_relaxed);
        context->framebuffer_dirty.store(true, std::memory_order_relaxed);
    }

    auto imgui_wants_keyboard() -> bool {
        return ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureKeyboard;
    }

    auto imgui_wants_mouse() -> bool {
        return ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureMouse;
    }

    auto viewport_has_mouse(Runtime const &app) -> bool {
        return app.viewport_hovered || app.mouse_dragging || app.game_mouse_captured;
    }

    auto key_callback(GLFWwindow *window, int key, int, int action, int mods) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr) {
            return;
        }

        if (action == GLFW_PRESS && !imgui_wants_keyboard()) {
            app->event_bus.publish(KeyPressedEvent{.key = key, .modifiers = mods});
        }
        if (action == GLFW_RELEASE) {
            app->event_bus.publish(KeyReleasedEvent{.key = key, .modifiers = mods});
        }
    }

    auto mouse_button_callback(GLFWwindow *window, int button, int action, int mods) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr) {
            return;
        }

        if (action == GLFW_RELEASE) {
            if (!app->is_playing && button == GLFW_MOUSE_BUTTON_RIGHT && app->mouse_dragging) {
                app->release_mouse();
            }

            app->event_bus.publish(MouseButtonReleasedEvent{.button = button, .modifiers = mods});
            return;
        }

        if (action != GLFW_PRESS || (imgui_wants_mouse() && !viewport_has_mouse(*app))) {
            return;
        }

        if (!app->is_playing) {
            if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                app->capture_mouse();
            }
        } else if (!app->play_fullscreen && app->viewport_hovered && !app->game_mouse_captured &&
                   !app->game->wants_cursor()) {
            app->game_mouse_captured = true;
            app->capture_mouse();
        }

        app->event_bus.publish(MouseButtonPressedEvent{.button = button, .modifiers = mods});
    }

    auto cursor_position_callback(GLFWwindow *window, double x_position, double y_position) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr) {
            return;
        }

        if (!app->has_last_mouse_position) {
            app->last_mouse_x = x_position;
            app->last_mouse_y = y_position;
            app->has_last_mouse_position = true;

            return;
        }

        auto const delta_x = x_position - app->last_mouse_x;
        auto const delta_y = y_position - app->last_mouse_y;

        app->last_mouse_x = x_position;
        app->last_mouse_y = y_position;

        app->event_bus.publish(MouseMovedEvent{.delta_x = delta_x, .delta_y = delta_y});
    }

    auto scroll_callback(GLFWwindow *window, double x_offset, double y_offset) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr || (imgui_wants_mouse() && !viewport_has_mouse(*app))) {
            return;
        }

        app->event_bus.publish(MouseScrolledEvent{.delta_x = x_offset, .delta_y = y_offset});
    }

    auto focus_callback(GLFWwindow *window, int focused) noexcept -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr) {
            return;
        }

        // The editor leaves play when the window loses focus; an installed game (--player) has no editor to fall back
        // into, so it carries on, which an online game needs to keep its connection and turn.
        if (focused == GLFW_FALSE && app->is_playing && !app->player_mode) {
            app->stop();
        }
    }

    auto drop_callback(GLFWwindow *window, int path_count, char const **paths) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr || path_count <= 0) {
            return;
        }

        std::vector<std::filesystem::path> dropped;
        dropped.reserve(static_cast<std::size_t>(path_count));

        for (auto const *path: std::span{paths, static_cast<std::size_t>(path_count)}) {
            dropped.push_back(gui::utf8_to_path(path));
        }

        app->on_files_dropped(dropped);
    }

    auto install_window_callbacks(VulkanContext &context, Runtime &app) noexcept -> void {
        static WindowData wd{};
        wd.app = &app;
        wd.ctx = &context;
        glfwSetWindowUserPointer(context.window, &wd);

        glfwSetFramebufferSizeCallback(context.window, framebuffer_size_callback);
        glfwSetKeyCallback(context.window, key_callback);
        glfwSetMouseButtonCallback(context.window, mouse_button_callback);
        glfwSetCursorPosCallback(context.window, cursor_position_callback);
        glfwSetScrollCallback(context.window, scroll_callback);
        glfwSetWindowFocusCallback(context.window, focus_callback);
        glfwSetDropCallback(context.window, drop_callback);
    }

    auto wait_idle_or_exit(VkDevice device, std::string_view label) noexcept -> VkResult {
        auto const result = wait_idle_bounded(device, label);

        if (result == VK_TIMEOUT) {
            std::_Exit(EXIT_FAILURE);
        }

        return result;
    }

    auto report_device_lost(VulkanContext &context) noexcept -> void {
        error("The GPU device was lost or stopped responding; the application cannot continue.");

        if (context.window != nullptr) {
            glfwHideWindow(context.window);
        }

        auto const shown = show_fatal_dialog(
                "Graphics device lost",
                "The GPU stopped responding, so the renderer cannot continue.\n\n"
                "Please restart the runtime. Work since the last save could not be recovered.\n\n"
                "If this keeps happening, update your graphics driver and check the log for details.");

        if (!shown) {
            error("No dialog tool is available to show the device-lost notice; see the log above.");
        }
    }

    auto destroy_runtime(VulkanContext &context, Runtime &runtime) noexcept -> void {
        // The streamers wait for every request they queued before they let go of the device; have the ones that have
        // not started give up, or an exit during a big load waits for the whole backlog.
        request_thread_pool_shutdown();

        // The GPU encoder lives on the device: stop its thread and release it first. Requests still waiting on it fail,
        // and the texture workers give up (the pool is shutting down) rather than compress them on the CPU.
        install_gpu_texture_service(nullptr);

        if (context.device != VK_NULL_HANDLE) {
            auto const result = wait_idle_or_exit(context.device, "destroy_runtime");

            if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST) {
                report_vk_error("vkDeviceWaitIdle(runtime destroy)", result);
            }
        }
        runtime.on_shutdown();

        runtime.debug_renderer.reset();
        runtime.imgui_renderer.reset();
        runtime.text_renderer.reset();
        runtime.ray_tracing_poc.destroy();
        runtime.renderer->destroy();

        context.destroy();
    }
}

namespace {

    [[nodiscard]] auto game_help() -> std::string {
        std::string names;
        for (auto const name: game_names()) {
            names += names.empty() ? "" : ", ";
            names += name;
        }
        return std::format("Which game to run: {} (default {})", names, game_names().front());
    }

    CVar<std::string> game_cvar{{
            .name = "game",
            .group = "Game",
            .help = "",
            .help_fn = game_help,
            .hint = "NAME",
            .validator = [](std::string_view text) -> std::expected<void, std::string> {
                auto const names = game_names();

                if (std::ranges::find(names, text) != names.end()) {
                    return {};
                }

                std::string expected;
                for (auto const name: names) {
                    expected += expected.empty() ? "" : "|";
                    expected += name;
                }
                return std::unexpected(std::format("'{}' (expected {})", text, expected));
            },
    }};

}

auto EngineProfile::save_scene(Runtime &runtime, std::filesystem::path const &path) -> void {
    (void) runtime;
    warn("Cannot cook '{}': --save-scene needs the editor", path.string());
}

static std::atomic<bool> g_running{true};
static auto ctrl_c_handler(int) -> void {
    g_running.store(false, std::memory_order_relaxed);
    glfwPostEmptyEvent();
}

// Models and textures stream in after startup, and a scene saved before they arrive would hold the stand-ins instead of
// the files. Try again next frame until nothing is loading.
static auto queue_save_scene_when_loaded(Runtime &runtime, EngineProfile &profile, std::filesystem::path path) -> void {
    runtime.renderer->queue_render_thread_event([&runtime, &profile, path = std::move(path)]() mutable {
        if (runtime.renderer->model_streamer().pending_count() + runtime.renderer->texture_streamer().pending_count() != 0) {
            queue_save_scene_when_loaded(runtime, profile, std::move(path));
            return;
        }
        profile.save_scene(runtime, path);
    });
}

auto run_engine(int argc, char **argv, EngineProfile &profile) -> int {
    CommandLine cli{std::string{profile.name()}, std::string{profile.description()}};

    // Weakest first: the environment, then the command line.
    for (auto const &problem: CVarRegistry::global().apply_environment()) {
        std::println(stderr, "{}: ignoring {}", profile.name(), problem);
    }

    CVarRegistry::global().attach(cli);
    profile.add_arguments(cli);

    auto const parsed = cli.parse(argc, argv);
    if (!parsed) {
        std::println(stderr, "{}: {}\nTry --help.", profile.name(), parsed.error());
        return EXIT_FAILURE;
    }
    if (*parsed == CommandLine::Outcome::help) {
        std::fputs(cli.help_text().c_str(), stdout);
        return EXIT_SUCCESS;
    }

    if (auto const exit_code = profile.early_exit()) {
        return *exit_code;
    }

    std::optional<ProjectFile> project;

    // A project is looked for only when nothing says where the data is: an installed game never is one.
    if (cvars::data_dir.get().empty()) {
        std::error_code ignored;
        auto const found = !cvars::project.get().empty() ? std::optional{std::filesystem::path{cvars::project.get()}}
                                                   : ProjectFile::find_root(std::filesystem::current_path(ignored));

        if (found) {
            if (auto loaded = ProjectFile::load(*found)) {
                project = std::move(*loaded);
            } else {
                error("Could not open the project: {}", loaded.error());

                if (!cvars::project.get().empty()) {
                    return EXIT_FAILURE;
                }
            }
        }
    }

    Paths::set_current(Paths::resolve({
            .data_dir = cvars::data_dir.get().empty() ? std::nullopt : std::optional{std::filesystem::path{cvars::data_dir.get()}},
            .project_root = project ? std::optional{project->root} : std::nullopt,
    }));
    std::optional<GameManifest> manifest;

    if (auto const manifest_path = Paths::current().data_root() / "game.toml"; std::filesystem::exists(manifest_path)) {
        if (auto loaded = GameManifest::load(manifest_path)) {
            manifest = std::move(*loaded);
        } else {
            error("Ignoring game manifest: {}", loaded.error());
        }
    }

    // Which game and script to start: a packaged game's manifest, else the project's. Only the manifest means player mode.
    auto const launch = manifest ? manifest : project ? std::optional{project->manifest} : std::nullopt;

    if (project) {
        info("Project '{}' at {}", project->manifest.name, project->root.string());
        RecentProjects::user_default().remember(project->root);
    }

    auto const player_mode = manifest.has_value() || cvars::player.get() || profile.always_player();

    if (!cvars::record_assets.get().empty()) {
        Paths::start_access_recording();
    }

    if (!cvars::record_resources.get().empty()) {
        start_resource_recording();
    } else if (auto const resource_pack_path = Paths::current().engine_root() / "resources.lbf";
               std::filesystem::exists(resource_pack_path)) {
        if (auto pack = ResourcePack::open(resource_pack_path)) {
            info("Using {} bundled resources from '{}'", (*pack)->size(), resource_pack_path.string());
            install_resource_provider(*std::move(pack));
        } else {
            error("Ignoring resource pack: {}", describe(pack.error()));
        }
    }

    if (cvars::clear_cache.get()) {
        auto const removed = caches::clear_all();

        info("Cleared the cache: {} entries ({})", removed.entries, human_readable_bytes(removed.bytes));
        return EXIT_SUCCESS;
    }

    if (cvars::cache_stats.get()) {
        info("{}", caches::report());
        return EXIT_SUCCESS;
    }

    caches::maintain(static_cast<std::uint64_t>(cvars::cache_max_size.get()) * 1024ULL * 1024ULL);

    if (!cvars::bake_shaders.get().empty()) {
        auto const baked = renderer::bake_shaders(Renderer::compiler(), cvars::bake_shaders.get());

        if (!baked) {
            error("{}", baked.error());
            return EXIT_FAILURE;
        }

        info("Baked {} entry points and {} variants to '{}'", baked->entry_points, baked->variants,
             cvars::bake_shaders.get());
        return EXIT_SUCCESS;
    }

    // A packaged pack beside the data wins; otherwise dev runs cache a pack per checkout (keyed by its engine root, so
    // separate checkouts don't evict each other) in the cache store.
    auto const shader_pack_cache = ShaderPackCache::global();
    auto const shader_pack_key = Paths::current().engine_root().string();
    auto const packaged_pack_path = Paths::current().engine_root() / "shaders.lsp";
    auto const default_pack_path = std::filesystem::exists(packaged_pack_path)
                                           ? packaged_pack_path
                                           : shader_pack_cache.find(shader_pack_key).value_or(std::filesystem::path{});
    auto cache_shaders = false;

    if (!cvars::record_shaders.get().empty()) {
        renderer::start_shader_recording();
    } else {
        auto const pack_path = cvars::shader_pack.get().empty() ? default_pack_path
                                                          : std::filesystem::path{cvars::shader_pack.get()};
        auto pack_usable = false;

        if (std::filesystem::exists(pack_path)) {
            if (auto pack = renderer::ShaderPack::load(pack_path)) {
                if (pack->matches_sources(Paths::current().engine_root() / "assets" / "shaders")) {
                    info("Using {} precompiled shaders from '{}'", pack->size(), pack_path.string());
                    renderer::install_shader_pack(std::make_shared<renderer::ShaderPack const>(std::move(*pack)));
                    pack_usable = true;
                } else {
                    warn("Ignoring shader pack '{}': the shader sources changed since it was recorded; "
                         "re-record it with --record-shaders",
                         pack_path.string());
                }
            } else {
                error("Ignoring shader pack: {}", pack.error());
            }
        }

        // No usable pack (first run, or the sources changed): record what this run compiles into the default
        // pack so the next run skips compilation. An explicit --shader-pack is never overwritten.
        if (!pack_usable && cvars::shader_pack.get().empty()) {
            renderer::start_shader_recording();
            cache_shaders = true;
        }
    }

    Renderer::prefetch_shaders();

    info("Starting GLFW Vulkan test, data directory {}", Paths::current().data_root().string());

    std::signal(SIGINT, ctrl_c_handler);

    auto const screen_type = cvars::screen_type.get();

    if (auto const prepared = profile.prepare(); !prepared) {
        error("{}", prepared.error());
        return EXIT_FAILURE;
    }

    set_texture_max_extent(cvars::max_texture_size.get());
    set_texture_rdo_lambda(cvars::texture_rdo.get());

    auto const dump_frame_graph = cvars::frame_graph_dump.get();

    VulkanContext context{};
    context.vsync = cvars::vsync.value_or(!profile.deterministic());
    context.async_compute_mode = cvars::async_compute.get();
    context.swapchain_image_count = cvars::swapchain_images.value_or(0U);
    context.hdr = cvars::hdr.value_or(!profile.deterministic());
    context.hdr_paper_white_nits = cvars::paper_white.get();
    context.hdr_peak_nits = std::max(cvars::peak_nits.get(), context.hdr_paper_white_nits);

    auto const chosen_present_mode = cvars::present_mode.has_value() ? cvars::present_mode.value()
                                     : (profile.deterministic() && !context.vsync)
                                             ? std::optional{PresentModeChoice::immediate}
                                             : std::nullopt;
    if (chosen_present_mode) {
        switch (*chosen_present_mode) {
            case PresentModeChoice::immediate:
                context.present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
                break;
            case PresentModeChoice::mailbox:
                context.present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
                break;
            case PresentModeChoice::fifo:
                context.present_mode = VK_PRESENT_MODE_FIFO_KHR;
                break;
            case PresentModeChoice::fifo_relaxed:
                context.present_mode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
                break;
        }
    }

#ifndef NDEBUG
    constexpr auto validation_by_default = true;
#else
    constexpr auto validation_by_default = false;
#endif
    context.validation = cvars::validation.value_or(validation_by_default);
    context.sync_validation = true;

    context.undecorated_window = profile.custom_title_bar() && screen_type == ScreenType::windowed;

    if (!initialize_vulkan(context, screen_type)) {
        error("Vulkan initialization failed");

        context.destroy();

        return EXIT_FAILURE;
    }

    if (!cvars::gpu_texture_bench.get().empty()) {
        auto const ok = run_gpu_texture_benchmark(context, Renderer::compiler(), cvars::gpu_texture_bench.get());

        context.destroy();

        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    if (cvars::gpu_texture_encoder.get()) {
        if (auto encoder = GpuTextureEncoder::create(context, Renderer::compiler())) {
            install_gpu_texture_service(GpuTextureService::with_encoder(std::move(*encoder)));
            info("Texture compression: BC7 on the GPU");
        } else {
            info("Texture compression: on the CPU ({})", encoder.error());
        }
    } else {
        info("Texture compression: on the CPU (--gpu-texture-encoder off)");
    }

    if (!context.hdr) {
        info("HDR output: off ({})", cvars::hdr.has_value() ? "--hdr off" : "deterministic run");
    } else if (context.swapchain.hdr_active()) {
        info("HDR output: on (scRGB, paper white {} nits, peak {} nits, metadata {})", context.hdr_paper_white_nits,
             context.hdr_peak_nits, context.hdr_metadata_supported ? "set" : "unsupported");
    } else {
        warn("HDR output: requested but unsupported by this surface, using sRGB");
    }

    auto const runtime_owner = profile.create_runtime(context);
    Runtime &runtime = *runtime_owner;
    runtime.player_mode = player_mode;
    runtime.project = project;
    runtime.screenshot_viewport_only = !cvars::screenshot_window.get();
    runtime.game_name = !game_cvar.get().empty() ? game_cvar.get()
                            : launch && !launch->game.empty() ? launch->game
                                                 : std::string{game_names().front()};
    runtime.script_entry = !cvars::script.get().empty() ? cvars::script.get() : launch ? launch->entry : std::string{};
    runtime.game_factory = [](std::string_view name) { return create_game(name); };
    runtime.available_games = game_names();
    runtime.game = create_game(runtime.game_name);
    runtime.subscribe_game_input();

    runtime.game->attach_host(GameHost{
            .player_mode = player_mode,
            .request_exit = [window = context.window] { glfwSetWindowShouldClose(window, GLFW_TRUE); },
            .script_entry = !cvars::script.get().empty() ? cvars::script.get() : launch ? launch->entry : std::string{},
            .effects = &runtime.effect_system,
            .text = runtime.text_renderer.get(),
    });

    if (launch && !launch->title.empty()) {
        glfwSetWindowTitle(context.window, launch->title.c_str());
    }
    install_window_callbacks(context, runtime);

    if (!initialize_runtime(context, runtime)) {
        destroy_runtime(context, runtime);
        return EXIT_FAILURE;
    }

    {
        auto mask = std::uint32_t{0};
        auto const automatic = cvars::async_passes.get().empty() || cvars::async_passes.get() == "auto";
        auto const names = automatic ? std::string{"gtao"} : cvars::async_passes.get();
        for (auto const name: std::views::split(names, ',')) {
            auto const token = std::string_view{name.begin(), name.end()};
            if (token == "light") {
                mask |= Renderer::async_light_clustering;
            } else if (token == "occlusion") {
                mask |= Renderer::async_occlusion;
            } else if (token == "gtao") {
                mask |= Renderer::async_gtao;
            } else if (token != "none" && token != "auto") {
                warn("--async-passes: unknown pass '{}' (expected light, occlusion, gtao or none)", token);
            }
        }
        runtime.renderer->set_async_candidates(mask);
        runtime.renderer->set_async_gtao_automatic(automatic);
    }
    runtime.renderer->set_frame_graph_dump(dump_frame_graph);
    runtime.renderer->set_render_scale(cvars::render_scale.get());
    if (cvars::max_quality.get()) {
        auto &renderer = *runtime.renderer;
        renderer.set_samples(VK_SAMPLE_COUNT_4_BIT);
        renderer.set_meshlet_culling(true);
        renderer.set_occlusion_culling(true);
        renderer.set_meshlet_occlusion_culling(true);
        renderer.set_clustered_lighting(true);
        renderer.set_decals_enabled(true);

        auto bloom = renderer.bloom_settings();
        bloom.enabled = true;
        renderer.set_bloom_settings(bloom);

        auto fog = renderer.fog_settings();
        fog.enabled = true;
        renderer.set_fog_settings(fog);

        auto ao = renderer.ao_settings();
        ao.enabled = true;
        ao.slice_count = 4;
        ao.step_count = 12;
        renderer.set_ao_settings(ao);

        auto local_shadows = renderer.local_shadow_settings();
        local_shadows.enabled = true;
        local_shadows.maximum_lights = 64;
        local_shadows.minimum_screen_radius_pixels = 4.0F;
        local_shadows.gpu_culling = true;
        local_shadows.caching = true;
        renderer.set_local_shadow_settings(local_shadows);

        info("--max-quality: MSAA up to 4x ({}x supported), fog, AO {}x{}, up to 64 shadowed local lights",
             static_cast<std::uint32_t>(renderer.max_samples()), ao.slice_count, ao.step_count);
    }
    runtime.ray_tracing_poc_enabled = cvars::rt_poc.get();
    {
        auto const names = RayTracingPoc::view_names;
        auto const found = std::ranges::find(names, std::string_view{cvars::rt_poc_view.get()});
        if (found == names.end()) {
            warn("--rt-poc-view '{}' is not a view; using beauty", cvars::rt_poc_view.get());
        } else {
            runtime.ray_tracing_poc.set_view(static_cast<std::uint32_t>(found - names.begin()));
        }
    }
    runtime.renderer->set_frame_graph_dot(cvars::frame_graph_dot.get());

    runtime.on_startup();

    // A packaged game ships its own cooked scene; it has to be open before play starts, because opening stops play.
    auto const manifest_scene = manifest && !manifest->scene.empty() && !cvars::scene.has_value()
                                        ? std::optional{Paths::current().data_root() / manifest->scene}
                                        : std::nullopt;

    if (manifest_scene) {
        runtime.renderer->queue_render_thread_event([&runtime, path = *manifest_scene] {
            runtime.play_when_loaded = true;
            runtime.open_scene(path);
        });
    } else if (player_mode) {
        runtime.renderer->queue_render_thread_event([&runtime] {
            runtime.play_fullscreen = true;
            runtime.play();
        });
    }

    // A project opens its main scene for editing, without starting play.
    if (project && !manifest && !cvars::scene.has_value() && !project->manifest.scene.empty()) {
        runtime.renderer->queue_render_thread_event(
                [&runtime, path = project->root / project->manifest.scene] { runtime.open_scene(path); });
    }

    if (auto const open_scene = cvars::scene.value()) {
        runtime.renderer->queue_render_thread_event(
                [&runtime, path = *open_scene] { runtime.open_scene(path); });
    }
    if (auto const save_scene = cvars::save_scene.value()) {
        queue_save_scene_when_loaded(runtime, profile, *save_scene);
    }

    if (auto const attached = profile.attach(runtime, context); !attached) {
        error("{}", attached.error());
        destroy_runtime(context, runtime);
        return EXIT_FAILURE;
    }

    FrameClock frame_clock;

    info("Initialization complete; close the window to exit");

    auto renderer_extent = context.swapchain.extent();
    constexpr auto resize_settle_time = std::chrono::milliseconds{150};
    auto pending_extent = renderer_extent;
    auto pending_since = std::chrono::steady_clock::now();
    auto last_frame_time = std::chrono::steady_clock::now();
    auto exit_code = EXIT_SUCCESS;

    std::uint32_t frames_run = 0;

    struct InjectedKey {
        std::uint32_t frame = 0;
        int key = 0;
        int modifiers = 0;
    };
    std::vector<InjectedKey> injected_keys;

    for (auto const item: CommandLine::split(cvars::inject_keys.get(), ',')) {
        auto const colon = item.find(':');

        if (colon == std::string_view::npos) {
            continue;
        }

        std::uint32_t frame = 0;
        int key = 0;
        int modifiers = 0;

        std::from_chars(item.data(), item.data() + colon, frame);

        auto spec = item.substr(colon + 1);

        for (auto const &[name, bit]: {std::pair{std::string_view{"ctrl+"}, GLFW_MOD_CONTROL},
                                       std::pair{std::string_view{"shift+"}, GLFW_MOD_SHIFT},
                                       std::pair{std::string_view{"alt+"}, GLFW_MOD_ALT}}) {
            if (spec.starts_with(name)) {
                modifiers |= bit;
                spec.remove_prefix(name.size());
            }
        }

        std::from_chars(spec.data(), spec.data() + spec.size(), key);

        injected_keys.push_back(InjectedKey{.frame = frame, .key = key, .modifiers = modifiers});
    }

    struct BenchPass {
        frame_graph::LogicalQueue queue = frame_graph::LogicalQueue::graphics;
        double total_ms = 0.0;
    };
    std::map<std::string, BenchPass> bench_passes;
    double bench_full_frame_ms = 0.0;
    std::uint32_t bench_samples = 0;
    std::uint64_t bench_last_serial = ~std::uint64_t{0};

    // A close request is handled right after polling, where the runtime may veto it (unsaved changes).
    while (g_running.load(std::memory_order_acquire) && context.running.load(std::memory_order_acquire)) {
        ZoneScopedNC("MainLoop", tracy::Color::Gray);

        auto const frame_number = frames_run++;

        if (cvars::exit_after_frames.get() != 0 && frame_number >= cvars::exit_after_frames.get()) {
            break;
        }

        if (cvars::screenshot_frame.get() != 0 && frame_number == cvars::screenshot_frame.get()) {
            runtime.request_screenshot();
        }

        for (auto const &injected: injected_keys) {
            if (injected.frame == frame_number) {
                runtime.event_bus.publish(KeyPressedEvent{.key = injected.key, .modifiers = injected.modifiers});
            }
        }

        frame_clock.begin_frame();

        auto const width = context.framebuffer_width.load(std::memory_order_relaxed);
        auto const height = context.framebuffer_height.load(std::memory_order_relaxed);

        {
            auto const polling = frame_clock.scope(CpuPhase::events);

            if (width <= 0 || height <= 0) {
                glfwWaitEvents();
            } else {
                glfwPollEvents();
            }
        }

        if (glfwWindowShouldClose(context.window) == GLFW_TRUE) {
            if (runtime.confirm_exit()) {
                break;
            }

            glfwSetWindowShouldClose(context.window, GLFW_FALSE);
        }

        {
            auto const draining = frame_clock.scope(CpuPhase::events);
            runtime.renderer->drain_event_queue();
        }

        auto const current_width = context.framebuffer_width.load(std::memory_order_relaxed);
        auto const current_height = context.framebuffer_height.load(std::memory_order_relaxed);

        if (current_width <= 0 || current_height <= 0) {
            last_frame_time = std::chrono::steady_clock::now();
            continue;
        }

        auto const now = std::chrono::steady_clock::now();
        // A benchmark advances the scene by frame, not by wall-clock time, so every run times the same frames.
        auto const measured_step = cvars::bench_frames.get() != 0 ? 1.0F / 60.0F
                                                            : std::chrono::duration<float>(now - last_frame_time).count();
        auto const delta_time = profile.fixed_timestep().value_or(measured_step);
        last_frame_time = now;

        {
            auto const updating = frame_clock.scope(CpuPhase::update);

            runtime.elapsed_time += delta_time;
            runtime.pre_update(delta_time);

            profile.begin_frame(runtime);

            runtime.update(delta_time);
        }

        request_resize_if_needed(context, current_width, current_height);

        if (!draw(context, runtime, frame_clock)) {
            exit_code = EXIT_FAILURE;
            break;
        }

        FrameMark;

        if (cvars::bench_pipeline_stats.get()) {
            runtime.renderer->request_pipeline_stats();
        }

        if (auto const &timings = runtime.renderer->last_frame_timings();
            timings.valid && runtime.can_start_recording_statistics()) {
            runtime.on_frame_timings(timings);

            if (cvars::bench_frames.get() != 0 && frame_number >= cvars::bench_warmup.get() &&
                timings.frame_serial != bench_last_serial) {
                bench_last_serial = timings.frame_serial;
                bench_full_frame_ms += timings.full_frame_ms;
                ++bench_samples;
                for (auto const &pass: timings.passes) {
                    auto &entry = bench_passes[pass.name_id];
                    entry.queue = pass.queue;
                    entry.total_ms += pass.milliseconds.value_or(0.0F);
                }
                if (bench_samples >= cvars::bench_frames.get()) {
                    std::println("[bench] {} frames after {} warm-up, mean GPU ms per frame:", bench_samples,
                                 cvars::bench_warmup.get());
                    auto pass_sum = 0.0;
                    for (auto const &[name, entry]: bench_passes) {
                        auto const mean = entry.total_ms / static_cast<double>(bench_samples);
                        pass_sum += mean;
                        std::println("[bench]   {:<32} {:>8.4f} ms  ({})", name, mean,
                                     entry.queue == frame_graph::LogicalQueue::graphics ? "graphics" : "compute");
                    }
                    std::println("[bench] sum of passes {:.4f} ms, full frame {:.4f} ms", pass_sum,
                                 bench_full_frame_ms / static_cast<double>(bench_samples));
                    break;
                }
            }
        }

        {
            auto const status = profile.end_frame(
                    runtime, context, frame_clock.finish_frame(),
                    RenderExtent{.width = renderer_extent.width, .height = renderer_extent.height});

            if (status == EngineProfile::LoopStatus::failed) {
                exit_code = EXIT_FAILURE;
                break;
            }
            if (status == EngineProfile::LoopStatus::finished) {
                break;
            }
        }

        auto const desired_render_extent = [&]() -> VkExtent2D {
            if (runtime.is_playing && runtime.play_fullscreen) {
                return context.swapchain.extent();
            }

            auto const &size = runtime.viewport_content_size;
            if (size.x <= 0.0F || size.y <= 0.0F) {
                return renderer_extent;
            }

            return VkExtent2D{
                    .width = static_cast<std::uint32_t>(size.x),
                    .height = static_cast<std::uint32_t>(size.y),
            };
        }();

        auto target_render_extent = desired_render_extent;

        {
            auto const size = profile.render_extent(
                    RenderExtent{.width = desired_render_extent.width, .height = desired_render_extent.height});
            target_render_extent = VkExtent2D{.width = size.width, .height = size.height};
        }

        // The render scale is applied last so a profile that fixes the size (the benchmarks) still gets it.
        if (auto const scale = runtime.renderer->render_scale(); scale != 1.0F) {
            constexpr auto max_dimension = 16384.0F;
            auto const scaled = [&](std::uint32_t value) {
                return static_cast<std::uint32_t>(std::clamp(std::round(static_cast<float>(value) * scale), 1.0F, max_dimension));
            };
            target_render_extent = VkExtent2D{.width = scaled(target_render_extent.width),
                                              .height = scaled(target_render_extent.height)};
        }

        if (!compare(target_render_extent, pending_extent)) {
            pending_extent = target_render_extent;
            pending_since = std::chrono::steady_clock::now();
        }

        auto const settled = profile.deterministic() ||
                             std::chrono::steady_clock::now() - pending_since >= resize_settle_time;

        if (settled && !compare(target_render_extent, renderer_extent)) {
            auto resize_result = runtime.renderer->resize(target_render_extent);

            if (!resize_result) {
                error("Could not resize renderer: {}", describe(resize_result.error()));

                exit_code = EXIT_FAILURE;
                break;
            }

            info("Resizing renderer to {}x{}", target_render_extent.width, target_render_extent.height);
            renderer_extent = target_render_extent;
        }

        runtime.update_views();
    }

    context.running.store(false, std::memory_order_release);

    if (!cvars::record_shaders.get().empty()) {
        if (auto const saved = renderer::finish_shader_recording(cvars::record_shaders.get())) {
            info("Recorded {} shaders to '{}'", *saved, cvars::record_shaders.get());
        } else {
            error("Could not write the shader pack: {}", saved.error());
            exit_code = EXIT_FAILURE;
        }
    }

    if (cache_shaders) {
        if (auto pending = shader_pack_cache.begin(shader_pack_key)) {
            if (auto const saved = renderer::finish_shader_recording(pending->path())) {
                info("Cached {} compiled shaders", *saved);
                static_cast<void>(shader_pack_cache.commit(
                        std::move(*pending), (Paths::current().engine_root() / "assets" / "shaders").string()));
            } else {
                warn("Could not cache compiled shaders: {}", saved.error());
            }
        }
    }

    if (!cvars::record_resources.get().empty()) {
        auto resources = finish_resource_recording();
        auto const count = resources.size();

        if (auto const written = ResourcePack::write(std::move(resources), cvars::record_resources.get())) {
            info("Recorded {} resources to '{}'", count, cvars::record_resources.get());
        } else {
            error("Could not write the resource pack: {}", describe(written.error()));
            exit_code = EXIT_FAILURE;
        }
    }

    if (!cvars::record_assets.get().empty()) {
        std::ofstream list{cvars::record_assets.get(), std::ios::trunc};
        auto const files = Paths::finish_access_recording();

        for (auto const &file: files) {
            list << file << '\n';
        }

        if (list) {
            info("Recorded {} data files to '{}'", files.size(), cvars::record_assets.get());
        } else {
            error("Could not write '{}'", cvars::record_assets.get());
            exit_code = EXIT_FAILURE;
        }
    }

    if (context.device_lost.load(std::memory_order_acquire)) {
        exit_code = EXIT_FAILURE;
        report_device_lost(context);
    }

    if (profile.interrupted()) {
        exit_code = EXIT_FAILURE;
    }

    destroy_runtime(context, runtime);

    if (exit_code == EXIT_SUCCESS) {
        info("Runtime exited successfully");
    }

    return exit_code;
}
