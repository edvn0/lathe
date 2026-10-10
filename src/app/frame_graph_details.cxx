#include "app/frame_graph_details.hxx"

#include <format>
#include <string>

#include <imgui.h>

#include "rendering/frame_graph/names.hxx"
#include "rendering/frame_graph/resource_info.hxx"
#include "rendering/imgui_renderer.hxx"
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

        auto preview_button(frame_graph::GraphDesc const &graph, std::uint32_t resource_index, Renderer &renderer)
                -> void {
            if (resource_index >= graph.resources.size() || !frame_graph::previewable(graph.resources[resource_index])) {
                return;
            }

            auto const &name = graph.resources[resource_index].name;
            if (renderer.preview_resource() == name) {
                ImGui::TextDisabled("previewing");
            } else if (ImGui::SmallButton("Preview")) {
                renderer.set_preview_resource(name);
            }
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

        auto draw_pass(frame_graph::FrameGraphView const &view, std::uint32_t pass_index, Renderer &renderer)
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

            if (ImGui::BeginTable("accesses", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Resource");
                ImGui::TableSetupColumn("Use");
                ImGui::TableSetupColumn("");
                ImGui::TableSetupColumn("Version");
                ImGui::TableSetupColumn("");
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
                    ImGui::TableNextColumn();
                    ImGui::PushID(static_cast<int>(access.resource));
                    preview_button(graph, access.resource, renderer);
                    ImGui::PopID();
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
                           Renderer &renderer) -> void {
        for (auto const pass: passes) {
            if (pass < view.graph.passes.size()) {
                draw_pass(view, pass, renderer);
            }
        }
    }

    namespace {

        auto pass_names(frame_graph::GraphDesc const &graph, std::vector<std::uint32_t> const &passes) -> std::string {
            auto text = std::string{};
            for (auto const pass: passes) {
                text += text.empty() ? "" : ", ";
                text += pass < graph.passes.size() ? graph.passes[pass].name : std::string{"?"};
            }
            return text.empty() ? std::string{"none"} : text;
        }

        auto mebibytes(std::uint64_t bytes) -> double { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

    }

    auto draw_resource_details(frame_graph::FrameGraphView const &view, std::span<std::uint32_t const> resources,
                               Renderer &renderer) -> void {
        auto const &graph = view.graph;

        for (auto const resource_index: resources) {
            if (resource_index >= graph.resources.size()) {
                continue;
            }

            auto const &resource = graph.resources[resource_index];
            auto const info = frame_graph::describe_resource(view, resource_index);

            ImGui::PushID(static_cast<int>(resource_index));
            ImGui::SeparatorText(resource.name.c_str());
            preview_button(graph, resource_index, renderer);

            auto const origin = resource.imported ? "imported" : "transient";
            if (resource.transient_image) {
                auto const &image = *resource.transient_image;
                auto const format = frame_graph::format_name(image.format);
                ImGui::Text("%s image, %.*s, %ux%u, %u mip%s, %ux samples", origin, static_cast<int>(format.size()),
                            format.data(), image.extent.width, image.extent.height, image.mip_levels,
                            image.mip_levels == 1 ? "" : "s", static_cast<unsigned>(image.samples));
            } else {
                ImGui::Text("%s %s", origin, resource.kind == frame_graph::ResourceKind::buffer ? "buffer" : "token");
            }

            ImGui::TextDisabled("Written by %s", pass_names(graph, info.producers).c_str());
            ImGui::TextDisabled("Read by %s", pass_names(graph, info.consumers).c_str());

            if (info.first_pass && info.last_pass) {
                ImGui::TextDisabled("Alive from %s to %s", graph.passes[*info.first_pass].name.c_str(),
                                    graph.passes[*info.last_pass].name.c_str());
            }

            if (info.placement) {
                ImGui::Text("%.2f MiB at offset %llu of block %u", mebibytes(info.placement->size),
                            static_cast<unsigned long long>(info.placement->offset), info.placement->block);

                if (!info.shares_memory_with.empty()) {
                    auto names = std::string{};
                    for (auto const other: info.shares_memory_with) {
                        names += names.empty() ? "" : ", ";
                        names += other < graph.resources.size() ? graph.resources[other].name : std::string{"?"};
                    }
                    ImGui::TextDisabled("Shares memory with %s", names.c_str());
                }
            }

            ImGui::PopID();
        }
    }

    auto previewed_resource(frame_graph::FrameGraphView const &view, Renderer const &renderer)
            -> std::optional<std::uint32_t> {
        if (renderer.preview_resource().empty()) {
            return std::nullopt;
        }

        for (auto index = std::size_t{0}; index < view.graph.resources.size(); ++index) {
            auto const &resource = view.graph.resources[index];
            if (resource.name == renderer.preview_resource() && frame_graph::previewable(resource)) {
                return static_cast<std::uint32_t>(index);
            }
        }
        return std::nullopt;
    }

    auto draw_preview(frame_graph::FrameGraphView const &view, std::uint32_t resource, Renderer &renderer) -> void {
        constexpr auto maximum_width = 320.0F;
        constexpr auto maximum_height = 170.0F;

        auto const &desc = view.graph.resources[resource];
        ImGui::TextUnformatted(desc.name.c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton("Close")) {
            renderer.set_preview_resource({});
            return;
        }

        auto const extent = desc.transient_image->extent;
        auto const aspect = static_cast<float>(extent.width) / static_cast<float>(std::max(extent.height, 1U));
        auto height = std::min(maximum_width / aspect, maximum_height);
        ImGui::Image(preview_texture_id(), ImVec2{height * aspect, height});
    }

}
