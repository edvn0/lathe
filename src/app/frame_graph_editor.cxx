#include "app/frame_graph_editor.hxx"

#include <algorithm>
#include <format>
#include <optional>
#include <string>

#include <imgui.h>
#include <imgui_node_editor.h>

#include "core/fly_string.hxx"
#include "rendering/renderer.hxx"

namespace ed = ax::NodeEditor;

namespace gui {
    namespace {

        constexpr auto link_id_tag = std::uintptr_t{1} << 62U;

        auto node_id_of(std::uintptr_t key) -> ed::NodeId { return ed::NodeId{key * 4}; }
        auto input_pin_of(std::uintptr_t key) -> ed::PinId { return ed::PinId{(key * 4) + 1}; }
        auto output_pin_of(std::uintptr_t key) -> ed::PinId { return ed::PinId{(key * 4) + 2}; }

        auto queue_name(frame_graph::LogicalQueue queue) -> char const * {
            return queue == frame_graph::LogicalQueue::graphics ? "graphics" : "compute";
        }

        auto pass_type_name(frame_graph::PassType type) -> char const * {
            switch (type) {
                case frame_graph::PassType::raster:
                    return "raster";
                case frame_graph::PassType::compute:
                    return "compute";
                case frame_graph::PassType::transfer:
                    return "transfer";
            }
            return "pass";
        }

        auto timing_of(Renderer const &renderer, std::string_view name_id) -> std::optional<float> {
            for (auto const &timing: renderer.frame_graph_timings()) {
                if (timing.name_id == name_id) {
                    return timing.milliseconds;
                }
            }
            return std::nullopt;
        }

    }

    struct FrameGraphEditor::Context {
        ed::EditorContext *editor = nullptr;

        Context() {
            auto config = ed::Config{};
            config.SettingsFile = nullptr;
            editor = ed::CreateEditor(&config);
        }

        ~Context() { ed::DestroyEditor(editor); }

        Context(Context const &) = delete;
        auto operator=(Context const &) -> Context & = delete;
        Context(Context &&) = delete;
        auto operator=(Context &&) -> Context & = delete;
    };

    FrameGraphEditor::FrameGraphEditor() = default;
    FrameGraphEditor::~FrameGraphEditor() = default;

    auto FrameGraphEditor::relayout(frame_graph::FrameGraphView const &view) -> void {
        layout_ = frame_graph::layout(view);
        laid_out_revision_ = view.revision;

        node_ids_.clear();
        node_ids_.reserve(view.graph.passes.size());
        for (auto const &pass: view.graph.passes) {
            node_ids_.push_back(FlyString{pass.name}.identity());
        }
        fit_countdown_ = placed_.empty() ? 6 : fit_countdown_;
    }

    auto FrameGraphEditor::draw(Renderer const &renderer, stages::Registry const &registry) -> void {
        auto const &view = renderer.frame_graph_view();
        if (view.revision == 0) {
            ImGui::TextDisabled("No frame graph yet");
            return;
        }

        if (!context_) {
            context_ = std::make_unique<Context>();
        }
        if (view.revision != laid_out_revision_) {
            relayout(view);
        }

        if (ImGui::Button("Fit")) {
            fit_countdown_ = 1;
        }

        ed::SetCurrentEditor(context_->editor);
        ed::Begin("frame_graph");

        for (auto const &node: layout_.nodes) {
            auto const &pass = view.graph.passes[node.pass];
            auto const key = node_ids_[node.pass];

            if (placed_.insert(key).second) {
                ed::SetNodePosition(node_id_of(key), ImVec2{node.x, node.y});
            }

            ed::BeginNode(node_id_of(key));
            ImGui::BeginGroup();

            if (node.culled) {
                ImGui::BeginDisabled();
            }

            auto const title = pass.profile.label.empty() ? pass.name : std::string{pass.profile.label};
            ImGui::TextUnformatted(title.c_str());

            if (auto const *stage = registry.stage_of_pass(FlyString{pass.name});
                stage != nullptr && stage->title != title) {
                ImGui::TextDisabled("%s", stage->title.data());
            }

            ImGui::TextDisabled("%s \xC2\xB7 %s%s", pass_type_name(pass.type), queue_name(node.queue),
                                node.culled ? " \xC2\xB7 culled" : "");

            if (auto const milliseconds = timing_of(renderer, pass.profile.name_id); milliseconds) {
                ImGui::Text("%.3f ms", static_cast<double>(*milliseconds));
            }

            ed::BeginPin(input_pin_of(key), ed::PinKind::Input);
            ImGui::TextUnformatted("in");
            ed::EndPin();
            ImGui::SameLine();
            ed::BeginPin(output_pin_of(key), ed::PinKind::Output);
            ImGui::TextUnformatted("out");
            ed::EndPin();

            if (node.culled) {
                ImGui::EndDisabled();
            }

            ImGui::EndGroup();
            ed::EndNode();
        }

        for (auto index = std::size_t{0}; index < layout_.edges.size(); ++index) {
            auto const &edge = layout_.edges[index];
            ed::Link(ed::LinkId{link_id_tag | index}, output_pin_of(node_ids_[edge.from]),
                     input_pin_of(node_ids_[edge.to]), ImVec4{0.62F, 0.68F, 0.78F, 0.45F}, 1.5F);
        }

        if (auto const hovered = ed::GetHoveredLink(); hovered) {
            auto const index = static_cast<std::size_t>(hovered.Get() & ~link_id_tag);
            if (index < layout_.edges.size()) {
                auto const &edge = layout_.edges[index];
                ed::Suspend();
                ImGui::SetTooltip("%s", view.graph.resources[edge.resource].name.c_str());
                ed::Resume();
            }
        }

        if (fit_countdown_ > 0 && --fit_countdown_ == 0) {
            ed::NavigateToContent();
        }

        ed::End();
        ed::SetCurrentEditor(nullptr);
    }

}
