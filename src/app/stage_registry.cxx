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

                auto const width = ImGui::GetContentRegionAvail().x;
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

    auto Registry::draw_settings(FlyString id, Renderer &renderer) -> void {
        if (auto const *stage = find(id); stage != nullptr && stage->draw_settings) {
            auto context = Context{.renderer = renderer, .ui = ui_};
            stage->draw_settings(context);
        }
    }

}
