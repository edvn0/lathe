#include "app/frame_graph_details.hxx"

#include <string>

#include <imgui.h>

#include "rendering/frame_graph/names.hxx"
#include "rendering/renderer.hxx"

namespace gui {
    namespace {

        auto resource_name(frame_graph::GraphDesc const &graph, std::uint32_t resource) -> char const * {
            return resource < graph.resources.size() ? graph.resources[resource].name.c_str() : "<unknown>";
        }

        auto op_name(frame_graph::OwnershipOp op) -> char const * {
            switch (op) {
                case frame_graph::OwnershipOp::none:
                    return "";
                case frame_graph::OwnershipOp::release:
                    return "queue release";
                case frame_graph::OwnershipOp::acquire:
                    return "queue acquire";
            }
            return "";
        }

        auto find_barriers(frame_graph::CompiledGraph const &compiled, std::uint32_t pass)
                -> frame_graph::BarrierSet const * {
            for (auto const &batch: compiled.batches) {
                for (auto const &compiled_pass: batch.passes) {
                    if (compiled_pass.pass == pass) {
                        return &compiled_pass.before;
                    }
                }
            }
            return nullptr;
        }

        auto draw_pass(frame_graph::FrameGraphView const &view, std::uint32_t pass_index, Renderer const &renderer)
                -> void {
            auto const &graph = view.graph;
            auto const &pass = graph.passes[pass_index];

            ImGui::PushID(static_cast<int>(pass_index));
            ImGui::SeparatorText(pass.name.c_str());

            for (auto const &timing: renderer.frame_graph_timings()) {
                if (timing.name_id == pass.profile.name_id && timing.milliseconds) {
                    ImGui::TextDisabled("%.3f ms", static_cast<double>(*timing.milliseconds));
                }
            }

            if (ImGui::BeginTable("accesses", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Resource");
                ImGui::TableSetupColumn("Use");
                ImGui::TableSetupColumn("");
                ImGui::TableSetupColumn("Version");
                ImGui::TableHeadersRow();

                for (auto const &access: pass.accesses) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(resource_name(graph, access.resource));
                    ImGui::TableNextColumn();
                    auto const use = frame_graph::use_name(access.use);
                    ImGui::TextUnformatted(use.data(), use.data() + use.size());
                    ImGui::TableNextColumn();
                    ImGui::TextDisabled("%s%s", access.produces ? "writes" : "reads", access.discard ? " (discard)" : "");
                    ImGui::TableNextColumn();
                    ImGui::Text("%u", access.version);
                }

                ImGui::EndTable();
            }

            if (auto const *barriers = find_barriers(view.compiled, pass_index);
                barriers != nullptr && !barriers->empty()) {
                ImGui::TextDisabled("Barriers before this pass");
                for (auto const &barrier: barriers->images) {
                    auto const from = frame_graph::layout_name(barrier.old_layout);
                    auto const to = frame_graph::layout_name(barrier.new_layout);
                    ImGui::BulletText("%s: %.*s -> %.*s %s", resource_name(graph, barrier.resource),
                                      static_cast<int>(from.size()), from.data(), static_cast<int>(to.size()),
                                      to.data(), op_name(barrier.op));
                }
                for (auto const &barrier: barriers->buffers) {
                    ImGui::BulletText("%s: buffer %s", resource_name(graph, barrier.resource), op_name(barrier.op));
                }
                if (!barriers->memory.empty()) {
                    ImGui::BulletText("%zu memory barrier%s", barriers->memory.size(),
                                      barriers->memory.size() == 1 ? "" : "s");
                }
            }

            ImGui::PopID();
        }

    }

    auto draw_pass_details(frame_graph::FrameGraphView const &view, std::span<std::uint32_t const> passes,
                           Renderer const &renderer) -> void {
        for (auto const pass: passes) {
            if (pass < view.graph.passes.size()) {
                draw_pass(view, pass, renderer);
            }
        }
    }

}
