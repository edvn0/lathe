#include "app/stage_registry.hxx"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>

#include "rendering/debug_renderer.hxx"
#include "rendering/hiz_occlusion.hxx"
#include "rendering/imgui_renderer.hxx"
#include "rendering/renderer.hxx"

namespace stages {
    namespace {

        auto draw_cluster_grid_settings(Renderer &renderer, std::optional<ClusterGridSettings> &refused) -> void {
            auto grid = refused.value_or(renderer.cluster_grid());
            bool changed = false;

            auto const preset = std::ranges::find(cluster_grid_presets, grid, &ClusterGridPreset::grid);
            std::string const preview =
                    preset == cluster_grid_presets.end() ? std::string{"Custom"} : std::string{preset->name};

            if (ImGui::BeginCombo("Cluster grid", preview.c_str())) {
                for (auto const &option: cluster_grid_presets) {
                    bool const selected = option.grid == grid;
                    auto const label =
                            std::format("{} ({}x{}x{}, {} per cluster)", option.name, option.grid.tiles_x,
                                        option.grid.tiles_y, option.grid.depth_slices, option.grid.light_capacity);

                    if (ImGui::Selectable(label.c_str(), selected)) {
                        grid = option.grid;
                        changed = true;
                    }
                    if (selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            auto const slider = [&](char const *label, std::uint32_t &value, std::uint32_t maximum,
                                    char const *tooltip, ImGuiSliderFlags flags = ImGuiSliderFlags_None) {
                auto edited = static_cast<int>(value);
                if (ImGui::SliderInt(label, &edited, 1, static_cast<int>(maximum), "%d",
                                     flags | ImGuiSliderFlags_AlwaysClamp)) {
                    value = static_cast<std::uint32_t>(edited);
                    changed = true;
                }
                ImGui::SetItemTooltip("%s", tooltip);
            };

            slider("Tiles across", grid.tiles_x, cluster_grid_maximum_tiles,
                   "Screen columns. More gives each pixel fewer lights to shade, at more build work.");
            slider("Tiles down", grid.tiles_y, cluster_grid_maximum_tiles,
                   "Screen rows. Tiles split the view evenly, so 9 rows to 16 columns keeps them square at 16:9.");
            slider("Depth slices", grid.depth_slices, cluster_grid_maximum_depth_slices,
                   "Exponential slices between the near and far planes. More separates lights at different depths.");
            slider("Lights per cluster", grid.light_capacity, cluster_grid_maximum_light_capacity,
                   "List capacity. A cluster touching more lights drops the highest-indexed ones (magenta in the "
                   "heatmap). Costs memory, not shading time.",
                   ImGuiSliderFlags_Logarithmic);

            if (changed) {
                if (auto applied = renderer.set_cluster_grid(grid); applied) {
                    refused.reset();
                } else {
                    refused = grid;
                }
            }

            auto const mebibytes = static_cast<double>(cluster_buffer_bytes(grid)) / (1024.0 * 1024.0);
            ImGui::TextDisabled("%u clusters, %.1f MiB per frame in flight", cluster_count(grid), mebibytes);

            if (refused) {
                auto const reason = validate_cluster_grid(*refused);
                ImGui::TextColored(ImVec4(1.0F, 0.45F, 0.40F, 1.0F), "Not applied: %s",
                                   reason ? "" : reason.error().c_str());
                return;
            }

            auto const &stats = renderer.last_cluster_stats();
            if (!stats.valid || stats.grid != grid) {
                return;
            }

            auto const total = cluster_count(stats.grid);
            auto const occupied_share = total == 0 ? 0.0 : 100.0 * stats.occupied_clusters / total;
            auto const average = stats.occupied_clusters == 0
                                         ? 0.0
                                         : static_cast<double>(stats.stored_lights) / stats.occupied_clusters;

            ImGui::Text("Occupied clusters: %u (%.0f%%)", stats.occupied_clusters, occupied_share);
            ImGui::Text("Lights per occupied cluster: %.1f average, %u most", average, stats.maximum_lights);

            if (stats.overflowing_clusters == 0) {
                ImGui::Text("Overflowing clusters: 0");
            } else {
                ImGui::TextColored(ImVec4(1.0F, 0.3F, 1.0F, 1.0F), "Overflowing clusters: %u",
                                   stats.overflowing_clusters);
                ImGui::SetItemTooltip("These drop lights. Raise Lights per cluster, or refine the grid so each "
                                      "cluster covers less of the scene.");
            }
        }

        auto draw_msaa(Context &context) -> void {
            auto &renderer = context.renderer;
            auto const max_samples = static_cast<std::uint32_t>(renderer.max_samples());
            auto const current = static_cast<std::uint32_t>(renderer.samples());
            auto const label = [](std::uint32_t count) {
                return count == 1 ? std::string{"Off"} : std::format("MSAA {}x", count);
            };

            if (ImGui::BeginCombo("MSAA", label(current).c_str())) {
                for (std::uint32_t count = 1; count <= max_samples; count <<= 1U) {
                    if (ImGui::Selectable(label(count).c_str(), count == current)) {
                        renderer.set_samples(static_cast<VkSampleCountFlagBits>(count));
                    }
                }

                ImGui::EndCombo();
            }
        }

        auto draw_hiz_preview(Context &context) -> void {
            auto &renderer = context.renderer;
            auto &hiz_debug_mip = context.ui.hiz_debug_mip;
            auto const pyramid_levels = renderer.hiz_debug_mip_count();

            if (pyramid_levels == 0) {
                return;
            }

            auto const top_mip = static_cast<int>(pyramid_levels) - 1;
            hiz_debug_mip = std::clamp(hiz_debug_mip, 0, top_mip);
            ImGui::SliderInt("Hi-Z mip", &hiz_debug_mip, 0, top_mip);

            auto const mip = static_cast<std::uint32_t>(hiz_debug_mip);

            if (auto const view = renderer.hiz_debug_view(mip); view.valid()) {
                auto const depth_extent = renderer.hiz_depth_extent();
                HizExtent const depth{.width = depth_extent.width, .height = depth_extent.height};
                auto const level = hiz_level_extent(depth, mip);
                auto const image = hiz_image_extent(depth);

                ImVec2 const uv_max{
                        static_cast<float>(level.width) / static_cast<float>(std::max(image.width >> mip, 1U)),
                        static_cast<float>(level.height) / static_cast<float>(std::max(image.height >> mip, 1U)),
                };

                auto width = ImGui::GetContentRegionAvail().x;
                if (context.max_preview_width > 0.0F) {
                    width = std::min(width, context.max_preview_width);
                }
                auto const height = width * static_cast<float>(depth.height) / static_cast<float>(depth.width);

                ImGui::Image(gui::linear_source_texture_id(view.index), ImVec2(width, height), ImVec2(0.0F, 0.0F),
                             uv_max);
                ImGui::TextDisabled("Red: each texel's farthest depth (reverse-Z: brighter is nearer)");
            }
        }

        auto draw_occlusion(Context &context) -> void {
            auto &renderer = context.renderer;

            bool meshlet_culling = renderer.meshlet_culling();
            if (ImGui::Checkbox("Meshlet culling (task shader)", &meshlet_culling)) {
                renderer.set_meshlet_culling(meshlet_culling);
            }

            bool const occlusion_supported = renderer.occlusion_culling_supported();
            bool occlusion_culling = renderer.occlusion_culling() && occlusion_supported;

            ImGui::BeginDisabled(!occlusion_supported);
            if (ImGui::Checkbox("Occlusion culling (Hi-Z, two-phase)", &occlusion_culling)) {
                renderer.set_occlusion_culling(occlusion_culling);
            }
            ImGui::EndDisabled();

            if (!occlusion_supported && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip(
                        "Unavailable: this device has no MIN depth resolve (VK_RESOLVE_MODE_MIN_BIT) for MSAA");
            }

            if (!occlusion_culling) {
                return;
            }

            ImGui::Indent();

            constexpr std::array occlusion_test_names{"Hi-Z", "Stub: never occluded", "Stub: always defer"};
            auto test_mode = static_cast<int>(renderer.occlusion_test_mode());

            if (ImGui::Combo("Occlusion test", &test_mode, occlusion_test_names.data(),
                             static_cast<int>(occlusion_test_names.size()))) {
                renderer.set_occlusion_test_mode(static_cast<OcclusionTestMode>(test_mode));
            }

            ImGui::SetItemTooltip("The stubs exercise the two-phase draw lists without the Hi-Z test: the frame must "
                                  "look exactly as with occlusion culling off.");

            bool meshlet_occlusion = renderer.meshlet_occlusion_culling();

            ImGui::BeginDisabled(!meshlet_culling);
            if (ImGui::Checkbox("Meshlet occlusion (task shader)", &meshlet_occlusion)) {
                renderer.set_meshlet_occlusion_culling(meshlet_occlusion);
            }
            ImGui::EndDisabled();

            if (!meshlet_culling && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Needs meshlet culling (task shader)");
            }

            draw_hiz_preview(context);

            ImGui::Unindent();
        }

        auto draw_clustered_lighting(Context &context) -> void {
            auto &renderer = context.renderer;

            bool clustered_lighting = renderer.clustered_lighting();
            if (ImGui::Checkbox("Clustered lighting (GPU)", &clustered_lighting)) {
                renderer.set_clustered_lighting(clustered_lighting);
            }

            ImGui::BeginDisabled(!clustered_lighting);
            bool cluster_heatmap = renderer.cluster_debug_heatmap();
            if (ImGui::Checkbox("Cluster light-count heatmap", &cluster_heatmap)) {
                renderer.set_cluster_debug_heatmap(cluster_heatmap);
            }
            draw_cluster_grid_settings(renderer, context.ui.refused_cluster_grid);
            ImGui::EndDisabled();

            ImGui::SeparatorText("Light LOD");
            auto light_lod = renderer.light_lod_settings();
            bool light_lod_dirty = false;
            light_lod_dirty |= ImGui::Checkbox("Screen-size light culling", &light_lod.enabled);
            ImGui::BeginDisabled(!light_lod.enabled);
            light_lod_dirty |=
                    ImGui::SliderFloat("Cull below (px)", &light_lod.cull_radius_pixels, 0.0F, 32.0F, "%.1f");
            ImGui::SetItemTooltip("On-screen radius of a light's range below which it is dropped before clustering.");
            light_lod_dirty |=
                    ImGui::SliderFloat("Full strength at (px)", &light_lod.fade_radius_pixels, 0.0F, 64.0F, "%.1f");
            ImGui::SetItemTooltip("Lights fade in between the cull radius and this one.");
            ImGui::EndDisabled();
            if (light_lod_dirty) {
                light_lod.fade_radius_pixels = std::max(light_lod.fade_radius_pixels, light_lod.cull_radius_pixels);
                renderer.set_light_lod_settings(light_lod);
            }
        }

        auto draw_shadows(Context &context) -> void {
            auto shadows = context.renderer.shadow_settings();
            bool dirty = false;

            dirty |= ImGui::SliderFloat("Split lambda", &shadows.cascades.split_lambda, 0.0F, 1.0F);
            dirty |= ImGui::SliderFloat("Shadow distance", &shadows.cascades.shadow_distance, 20.0F, 500.0F);
            dirty |= ImGui::SliderFloat("PCF radius", &shadows.pcf_radius_texels, 0.5F, 4.0F);
            dirty |= ImGui::SliderFloat("Normal offset", &shadows.normal_offset_texels, 0.0F, 8.0F);
            dirty |= ImGui::SliderFloat("Depth bias", &shadows.depth_bias_world, 0.0F, 0.5F);
            dirty |= ImGui::SliderFloat("Bias slope", &shadows.depth_bias_slope, -8.0F, 0.0F);
            dirty |= ImGui::Checkbox("Cascade tint", &shadows.debug_cascade_tint);

            if (dirty) {
                context.renderer.set_shadow_settings(shadows);
            }
        }

        auto draw_ambient_occlusion(Context &context) -> void {
            auto ao = context.renderer.ao_settings();
            bool dirty = false;

            dirty |= ImGui::Checkbox("Ambient occlusion (GTAO)", &ao.enabled);

            ImGui::BeginDisabled(!ao.enabled);
            dirty |= ImGui::SliderFloat("Radius", &ao.radius, 0.05F, 3.0F);
            dirty |= ImGui::SliderFloat("Falloff range", &ao.falloff_range, 0.05F, 1.0F);
            dirty |= ImGui::SliderFloat("Intensity", &ao.intensity, 0.0F, 2.0F);

            auto slices = static_cast<int>(ao.slice_count);
            if (ImGui::SliderInt("Slices", &slices, 1, 8)) {
                ao.slice_count = static_cast<std::uint32_t>(slices);
                dirty = true;
            }

            auto steps = static_cast<int>(ao.step_count);
            if (ImGui::SliderInt("Steps", &steps, 1, 16)) {
                ao.step_count = static_cast<std::uint32_t>(steps);
                dirty = true;
            }

            dirty |= ImGui::SliderFloat("Denoise depth sigma", &ao.denoise_depth_sigma, 1.0F, 200.0F, "%.1f",
                                        ImGuiSliderFlags_Logarithmic);
            ImGui::EndDisabled();

            if (dirty) {
                context.renderer.set_ao_settings(ao);
            }
        }

        auto draw_bloom(Context &context) -> void {
            auto bloom = context.renderer.bloom_settings();
            bool dirty = false;

            dirty |= ImGui::Checkbox("Bloom", &bloom.enabled);

            ImGui::BeginDisabled(!bloom.enabled);
            dirty |= ImGui::SliderFloat("Threshold", &bloom.threshold, 0.0F, 8.0F);
            dirty |= ImGui::SliderFloat("Knee", &bloom.knee, 0.0F, 2.0F);
            dirty |= ImGui::SliderFloat("Filter radius", &bloom.filter_radius, 0.25F, 4.0F);
            dirty |= ImGui::SliderFloat("Intensity", &bloom.intensity, 0.0F, 1.0F);
            ImGui::EndDisabled();

            if (dirty) {
                context.renderer.set_bloom_settings(bloom);
            }
        }

        auto draw_debug_overlays(Context &context) -> void {
            bool draw_light_icons = context.renderer.debug_draw_light_icons();
            if (ImGui::Checkbox("Draw light icons", &draw_light_icons)) {
                context.renderer.set_debug_draw_light_icons(draw_light_icons);
            }

            auto *debug = context.debug_renderer;
            if (debug == nullptr) {
                return;
            }

            bool draw_physics_debug = debug->physics_debug_enabled();
            if (ImGui::Checkbox("Draw physics colliders", &draw_physics_debug)) {
                debug->set_physics_debug_enabled(draw_physics_debug);
            }

            bool draw_model_bounds_debug = debug->model_bounds_debug_enabled();
            if (ImGui::Checkbox("Draw model submesh bounds", &draw_model_bounds_debug)) {
                debug->set_model_bounds_debug_enabled(draw_model_bounds_debug);
            }
        }

        auto passes_of(std::initializer_list<std::string_view> names) -> std::vector<FlyString> {
            auto passes = std::vector<FlyString>{};
            passes.reserve(names.size());
            for (auto const name: names) {
                passes.emplace_back(name);
            }
            return passes;
        }

    }

    Registry::Registry() {
        add(Stage{
                .id = FlyString{"msaa"},
                .title = "Anti-aliasing",
                .passes = passes_of({"depth_prepass", "forward_pass"}),
                .state =
                        [](Renderer const &renderer) {
                            return State{.enabled = renderer.samples() != VK_SAMPLE_COUNT_1_BIT};
                        },
                .set_enabled = {},
                .draw_settings = draw_msaa,
        });

        add(Stage{
                .id = FlyString{"occlusion_culling"},
                .title = "Culling",
                .passes = passes_of({"gpu_culling", "meshlet_visibility_clear", "hiz_build", "occlusion_culling",
                                     "depth_prepass_late", "occlusion_stats_clear", "occlusion_stats_readback"}),
                .state =
                        [](Renderer const &renderer) {
                            auto const supported = renderer.occlusion_culling_supported();
                            return State{
                                    .enabled = renderer.occlusion_culling() && supported,
                                    .supported = supported,
                                    .reason = supported ? std::string_view{} : "No MIN depth resolve for MSAA",
                            };
                        },
                .set_enabled = [](Renderer &renderer, bool enabled) { renderer.set_occlusion_culling(enabled); },
                .draw_settings = draw_occlusion,
        });

        add(Stage{
                .id = FlyString{"clustered_lighting"},
                .title = "Clustered lighting",
                .passes = passes_of({"cluster_stats_clear", "light_cull", "light_cluster", "cluster_stats_readback"}),
                .state = [](Renderer const &renderer) { return State{.enabled = renderer.clustered_lighting()}; },
                .set_enabled = [](Renderer &renderer, bool enabled) { renderer.set_clustered_lighting(enabled); },
                .draw_settings = draw_clustered_lighting,
        });

        add(Stage{
                .id = FlyString{"debug_overlays"},
                .title = "Debug overlays",
                .passes = passes_of({"overlay_prepare"}),
                .state = [](Renderer const &) { return State{}; },
                .set_enabled = {},
                .draw_settings = draw_debug_overlays,
        });

        add(Stage{
                .id = FlyString{"shadows"},
                .title = "Shadows",
                .passes = passes_of({"shadow_pass"}),
                .state = [](Renderer const &) { return State{}; },
                .set_enabled = {},
                .draw_settings = draw_shadows,
        });

        add(Stage{
                .id = FlyString{"ambient_occlusion"},
                .title = "Ambient occlusion",
                .passes = passes_of({"gtao", "gtao_denoise"}),
                .state = [](Renderer const &renderer) { return State{.enabled = renderer.ao_settings().enabled}; },
                .set_enabled =
                        [](Renderer &renderer, bool enabled) {
                            auto ao = renderer.ao_settings();
                            ao.enabled = enabled;
                            renderer.set_ao_settings(ao);
                        },
                .draw_settings = draw_ambient_occlusion,
        });

        add(Stage{
                .id = FlyString{"bloom"},
                .title = "Bloom",
                .passes = passes_of({"bloom"}),
                .state = [](Renderer const &renderer) { return State{.enabled = renderer.bloom_settings().enabled}; },
                .set_enabled =
                        [](Renderer &renderer, bool enabled) {
                            auto bloom = renderer.bloom_settings();
                            bloom.enabled = enabled;
                            renderer.set_bloom_settings(bloom);
                        },
                .draw_settings = draw_bloom,
        });
    }

    auto Registry::add(Stage stage) -> void {
        auto const index = stages_.size();
        by_id_.emplace(stage.id, index);
        for (auto const pass: stage.passes) {
            by_pass_.emplace(pass, index);
        }
        stages_.push_back(std::move(stage));
    }

    auto Registry::find(FlyString id) const -> Stage const * {
        auto const found = by_id_.find(id);
        return found == by_id_.end() ? nullptr : &stages_[found->second];
    }

    auto Registry::stage_of_pass(FlyString pass) const -> Stage const * {
        auto const found = by_pass_.find(pass);
        return found == by_pass_.end() ? nullptr : &stages_[found->second];
    }

    auto Registry::draw_settings(FlyString id, Renderer &renderer, float max_preview_width) -> void {
        if (auto const *stage = find(id); stage != nullptr && stage->draw_settings) {
            auto context = Context{.renderer = renderer, .ui = ui_, .max_preview_width = max_preview_width,
                                   .debug_renderer = debug_renderer_};
            stage->draw_settings(context);
        }
    }

}
