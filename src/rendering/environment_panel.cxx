#include "rendering/environment_panel.hxx"

#include <algorithm>
#include <array>
#include <format>

#include <imgui.h>

#include "rendering/imgui_renderer.hxx"

namespace gui {

    namespace {
        constexpr std::array<char const *, 3> source_names{"Flat ambient", "Procedural sky", "HDR image"};

        auto draw_status(EnvironmentStatus const &status) -> void {
            switch (status.phase) {
                case EnvironmentPhase::idle:
                    ImGui::TextDisabled("Idle");
                    break;
                case EnvironmentPhase::decoding:
                    ImGui::TextUnformatted(status.message.c_str());
                    break;
                case EnvironmentPhase::building:
                    ImGui::Text("Building");
                    if (status.step_count != 0) {
                        ImGui::SameLine();
                        ImGui::ProgressBar(static_cast<float>(status.step) / static_cast<float>(status.step_count),
                                           ImVec2(-1.0F, 0.0F));
                    }
                    break;
                case EnvironmentPhase::ready:
                    ImGui::TextColored(ImVec4(0.4F, 0.9F, 0.4F, 1.0F), "Ready");
                    break;
                case EnvironmentPhase::failed:
                    ImGui::TextColored(ImVec4(1.0F, 0.4F, 0.4F, 1.0F), "Failed: %s", status.message.c_str());
                    break;
            }
        }

        auto draw_faces(EnvironmentSystem const &system, bool prefilter, int mip) -> void {
            auto const available = ImGui::GetContentRegionAvail().x;
            auto const size = std::max((available - (5.0F * ImGui::GetStyle().ItemSpacing.x)) / 6.0F, 16.0F);

            for (std::uint32_t face = 0; face < 6; ++face) {
                auto const slot = prefilter ? system.prefilter_face_texture_index(static_cast<std::uint32_t>(mip), face)
                                            : system.radiance_face_texture_index(static_cast<std::uint32_t>(mip), face);

                if (slot != 0) {
                    ImGui::Image(linear_source_texture_id(slot), ImVec2(size, size));
                }

                if (face != 5) {
                    ImGui::SameLine();
                }
            }
        }
    } // namespace

    auto environment_file_filters() -> std::vector<FileBrowser::Filter> {
        return {
                {.label = "Environments (*.hdr, *.exr, *.ktx2)", .extensions = {".hdr", ".exr", ".ktx2"}},
                {.label = "All files", .extensions = {}},
        };
    }

    auto draw_environment_panel(SceneEnvironment &environment, EnvironmentSystem &system, FileBrowser &browser,
                                bool &browsing) -> bool {
        bool changed = false;

        auto const status = system.status();

        int source = static_cast<int>(environment.source);

        if (ImGui::Combo("Source", &source, source_names.data(), static_cast<int>(source_names.size()))) {
            environment.source = static_cast<EnvironmentSource>(source);
            changed = true;
        }

        draw_status(status);

        auto const procedural = environment.source == EnvironmentSource::procedural_sky;
        auto const hdr = environment.source == EnvironmentSource::hdr_image;

        if (hdr) {
            ImGui::SeparatorText("HDR image");

            ImGui::TextWrapped("%s", environment.hdr_source.empty() ? "(none)" : environment.hdr_source.c_str());

            ImGui::BeginDisabled(browser.is_open());
            if (ImGui::Button("Browse...")) {
                browser.open("Load Environment", environment_file_filters());
                browsing = true;
            }
            ImGui::EndDisabled();

            int size_index = environment.hdr_cube_size >= 1024 ? 1 : 0;
            if (ImGui::Combo("Cube size", &size_index, "512\0001024\0")) {
                environment.hdr_cube_size = size_index == 1 ? 1024U : 512U;
                changed = true;
            }
            ImGui::SetItemTooltip("Face size of the radiance cube for an equirect source; a cubemap keeps its own.");

            changed |= ImGui::SliderFloat("Rotation", &environment.rotation_degrees, -180.0F, 180.0F, "%.1f deg");
        }

        if (procedural) {
            ImGui::SeparatorText("Sky");

            changed |= ImGui::SliderFloat("Turbidity", &environment.sun.turbidity, 2.0F, 10.0F);
            ImGui::SetItemTooltip("Preetham is calibrated for about 2-6; hazier skies are extrapolated.");
            changed |= ImGui::ColorEdit3("Ground albedo", &environment.sun.ground_albedo.x);
            changed |= ImGui::SliderFloat("Sky intensity", &environment.sky_intensity, 0.0F, 4.0F);
            changed |= ImGui::SliderFloat("Sun radius", &environment.sun.angular_radius_degrees, 0.05F, 2.0F, "%.2f deg");
        }

        ImGui::SeparatorText("Sun");

        changed |= ImGui::SliderFloat("Azimuth", &environment.sun.azimuth_degrees, -180.0F, 180.0F, "%.1f deg");
        // Procedural skies follow the sun below the horizon; the light itself is held above 5 degrees for the cascades.
        changed |= ImGui::SliderFloat("Elevation", &environment.sun.elevation_degrees, procedural ? -10.0F : 5.0F, 89.0F,
                                      "%.1f deg");

        if (procedural) {
            changed |= ImGui::Checkbox("Derive colour from sky", &environment.sun.derive_colour_from_sky);
        }

        changed |= ImGui::ColorEdit3("Colour", &environment.sun.colour.x);
        changed |= ImGui::SliderFloat("Intensity", &environment.sun.intensity, 0.0F, 10.0F);
        changed |= ImGui::Checkbox("Drive directional light", &environment.sun_drives_directional_light);

        if (hdr && environment.sun_drives_directional_light && environment.sun.intensity > 0.0F) {
            ImGui::TextColored(ImVec4(1.0F, 0.8F, 0.3F, 1.0F),
                               "An HDRI with a baked sun double-counts it with the directional light.");
        }

        ImGui::SeparatorText("Lighting");

        if (environment.source == EnvironmentSource::flat_ambient) {
            changed |= ImGui::SliderFloat("Ambient intensity", &environment.ambient_intensity, 0.0F, 1.0F);
        } else {
            changed |= ImGui::SliderFloat("Exposure (EV)", &environment.exposure_ev, -6.0F, 6.0F);
            changed |= ImGui::SliderFloat("Diffuse IBL", &environment.diffuse_intensity, 0.0F, 4.0F);
            changed |= ImGui::SliderFloat("Specular IBL", &environment.specular_intensity, 0.0F, 4.0F);
            changed |= ImGui::SliderFloat("Specular occlusion", &environment.specular_occlusion, 0.0F, 1.0F);
            changed |= ImGui::Checkbox("Multi-scatter", &environment.multi_scatter);
            ImGui::SetItemTooltip("Energy compensation for rough specular, from the BRDF LUT.");
            changed |= ImGui::Checkbox("Draw skybox", &environment.draw_skybox);
            changed |= ImGui::Checkbox("Fog sky", &environment.fog_sky);
        }

        ImGui::SeparatorText("Fog");

        changed |= ImGui::Checkbox("Enabled", &environment.fog.enabled);
        changed |= ImGui::ColorEdit3("Fog colour", &environment.fog.colour.x);
        changed |= ImGui::SliderFloat("Fog extinction", &environment.fog.extinction, 0.0F, 0.02F, "%.4f");
        changed |= ImGui::SliderFloat("Fog inscattering", &environment.fog.inscattering, 0.0F, 2.0F);
        changed |= ImGui::Checkbox("Colour from environment", &environment.fog.from_environment);
        ImGui::SetItemTooltip("Tints the fog with the blurred environment, so it follows sunsets and HDRIs.");

        if (ImGui::CollapsingHeader("Debug")) {
            auto &debug = system.debug_settings();

            auto const view_count = static_cast<int>(EnvironmentDebugView::count);
            auto const current = static_cast<int>(debug.view);

            if (ImGui::BeginCombo("View", to_string(debug.view).data())) {
                for (int index = 0; index < view_count; ++index) {
                    auto const view = static_cast<EnvironmentDebugView>(index);

                    if (ImGui::Selectable(to_string(view).data(), index == current)) {
                        debug.view = view;
                    }
                }

                ImGui::EndCombo();
            }

            ImGui::SliderFloat("Prefilter LOD", &debug.prefilter_lod, 0.0F,
                               static_cast<float>(EnvironmentSystem::prefilter_mip_count() - 1));
            ImGui::Checkbox("Amortize rebuilds", &debug.amortize_rebuilds);

            if (ImGui::Button("Rebuild now")) {
                system.rebuild();
            }

            ImGui::SameLine();

            static std::string validation;

            if (ImGui::Button("Validate against CPU")) {
                validation = system.validate_against_cpu().summary;
            }

            if (!validation.empty()) {
                ImGui::TextWrapped("%s", validation.c_str());
            }

            static int radiance_mip = 0;
            static int prefilter_mip = 0;

            if (system.radiance_mip_count() != 0) {
                ImGui::Text("Radiance cube");
                ImGui::SliderInt("Radiance mip", &radiance_mip, 0, static_cast<int>(system.radiance_mip_count()) - 1);
                draw_faces(system, false, radiance_mip);
            }

            ImGui::Text("Prefilter cube");
            ImGui::SliderInt("Prefilter mip", &prefilter_mip, 0,
                             static_cast<int>(EnvironmentSystem::prefilter_mip_count()) - 1);
            draw_faces(system, true, prefilter_mip);

            ImGui::Text("BRDF LUT");
            if (system.brdf_lut_texture_index() != 0) {
                ImGui::Image(linear_source_texture_id(system.brdf_lut_texture_index()), ImVec2(128.0F, 128.0F));
            }
        }

        return changed;
    }

} // namespace gui
