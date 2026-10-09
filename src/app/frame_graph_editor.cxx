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
        constexpr auto stage_base_height = 90.0F;
        constexpr auto stage_settings_height = 300.0F;
        constexpr auto settings_item_width = 260.0F;

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

        auto total_milliseconds(Renderer const &renderer) -> float {
            auto total = 0.0F;
            for (auto const &timing: renderer.frame_graph_timings()) {
                total += timing.milliseconds.value_or(0.0F);
            }
            return total;
        }

        auto barrier_total(frame_graph::BarrierSet const &set) -> std::uint32_t {
            return static_cast<std::uint32_t>(set.images.size() + set.buffers.size() + set.memory.size());
        }

        auto stage_height(stages::Stage const &stage) -> float {
            return stage_base_height + (stage.draw_settings ? stage_settings_height : 0.0F);
        }

        // Grey to amber to red with the share of the frame a node takes.
        auto heat_colour(float share) -> ImVec4 {
            auto const t = std::clamp(share * 3.0F, 0.0F, 1.0F);
            return ImVec4{0.55F + (0.45F * t), 0.58F - (0.18F * t), 0.62F - (0.45F * t), 1.0F};
        }

        struct NodeStats {
            float milliseconds = 0.0F;
            std::uint32_t barriers = 0;
        };

        struct HeatScope {
            explicit HeatScope(float share) {
                ed::PushStyleColor(ed::StyleColor_NodeBorder, heat_colour(share));
                ed::PushStyleVar(ed::StyleVar_NodeBorderWidth, 1.5F + (4.0F * std::min(share * 3.0F, 1.0F)));
            }
            ~HeatScope() {
                ed::PopStyleVar();
                ed::PopStyleColor();
            }
            HeatScope(HeatScope const &) = delete;
            auto operator=(HeatScope const &) -> HeatScope & = delete;
            HeatScope(HeatScope &&) = delete;
            auto operator=(HeatScope &&) -> HeatScope & = delete;
        };

        auto draw_pins(std::uintptr_t key) -> void {
            ed::BeginPin(input_pin_of(key), ed::PinKind::Input);
            ImGui::TextUnformatted("in");
            ed::EndPin();
            ImGui::SameLine();
            ed::BeginPin(output_pin_of(key), ed::PinKind::Output);
            ImGui::TextUnformatted("out");
            ed::EndPin();
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

    auto FrameGraphEditor::reset_layout() -> void {
        placed_.clear();
        layout_dirty_ = true;
        fit_countdown_ = 6;
    }

    auto FrameGraphEditor::relayout(frame_graph::FrameGraphView const &view, stages::Registry const &registry)
            -> void {
        auto const stages = registry.stages();
        auto group_of_pass = std::vector<std::uint32_t>(view.graph.passes.size(), frame_graph::no_group);
        auto group_heights = std::vector<float>(stages.size());

        for (auto index = std::size_t{0}; index < stages.size(); ++index) {
            auto const measured = measured_height_.find(stages[index].id.identity());
            group_heights[index] = measured != measured_height_.end() ? measured->second : stage_height(stages[index]);
        }

        if (!expanded_) {
            for (auto pass = std::size_t{0}; pass < view.graph.passes.size(); ++pass) {
                if (auto const *stage = registry.stage_of_pass(FlyString{view.graph.passes[pass].name});
                    stage != nullptr) {
                    group_of_pass[pass] = static_cast<std::uint32_t>(stage - stages.data());
                }
            }
        }

        layout_ = frame_graph::layout(view, {}, {.group_of_pass = group_of_pass, .group_heights = group_heights});
        laid_out_revision_ = view.revision;
        layout_dirty_ = false;

        node_keys_.clear();
        node_keys_.reserve(layout_.nodes.size());
        auto present = std::vector<bool>(stages.size(), false);
        auto bottom = 0.0F;

        for (auto const &node: layout_.nodes) {
            if (node.group != frame_graph::no_group) {
                node_keys_.push_back(stages[node.group].id.identity());
                present[node.group] = true;
            } else {
                node_keys_.push_back(FlyString{view.graph.passes[node.passes.front()].name}.identity());
            }

            auto const height = node.group != frame_graph::no_group ? group_heights[node.group]
                                                                    : frame_graph::LayoutParams{}.default_height;
            bottom = std::max(bottom, node.y + height);
        }

        pass_barriers_.assign(view.graph.passes.size(), 0);
        for (auto const &batch: view.compiled.batches) {
            for (auto const &pass: batch.passes) {
                pass_barriers_[pass.pass] = barrier_total(pass.before);
            }
        }

        ghosts_.clear();
        if (!expanded_) {
            auto const params = frame_graph::LayoutParams{};
            auto column = 0.0F;
            for (auto index = std::size_t{0}; index < stages.size(); ++index) {
                if (!present[index]) {
                    ghosts_.push_back({.stage = index,
                                       .key = stages[index].id.identity(),
                                       .x = column * params.column_width,
                                       .y = bottom + params.lane_gap});
                    column += 1.0F;
                }
            }
        }
    }

    auto FrameGraphEditor::draw(Renderer &renderer, stages::Registry &registry) -> void {
        auto const &view = renderer.frame_graph_view();
        if (view.revision == 0) {
            ImGui::TextDisabled("No frame graph yet");
            return;
        }

        if (!context_) {
            context_ = std::make_unique<Context>();
        }

        if (ImGui::Button("Fit")) {
            fit_countdown_ = 1;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset layout")) {
            reset_layout();
        }
        ImGui::SameLine();
        if (ImGui::Checkbox("Show every pass", &expanded_)) {
            layout_dirty_ = true;
            fit_countdown_ = 6;
        }

        if (view.revision != laid_out_revision_ || layout_dirty_) {
            auto const first = placed_.empty();
            relayout(view, registry);
            settle_countdown_ = first ? 3 : settle_countdown_;
            fit_countdown_ = first ? 0 : fit_countdown_;
        }

        if (settle_countdown_ > 0 && --settle_countdown_ == 0) {
            reset_layout();
            relayout(view, registry);
        }

        auto const total_ms = std::max(total_milliseconds(renderer), 0.001F);
        auto const stage_list = registry.stages();

        ed::SetCurrentEditor(context_->editor);
        ed::Begin("frame_graph");

        for (auto index = std::size_t{0}; index < layout_.nodes.size(); ++index) {
            auto const &node = layout_.nodes[index];
            auto const key = node_keys_[index];

            if (placed_.insert(key).second) {
                ed::SetNodePosition(node_id_of(key), ImVec2{node.x, node.y});
            }

            auto stats = NodeStats{};
            for (auto const pass: node.passes) {
                stats.milliseconds += timing_of(renderer, view.graph.passes[pass].profile.name_id).value_or(0.0F);
                stats.barriers += pass_barriers_[pass];
            }

            auto const heat = HeatScope{stats.milliseconds / total_ms};
            ed::BeginNode(node_id_of(key));
            ImGui::BeginGroup();

            if (node.group != frame_graph::no_group) {
                auto const &stage = stage_list[node.group];
                ImGui::PushID(stage.id.c_str());

                auto const state = stage.state(renderer);
                if (stage.set_enabled) {
                    auto enabled = state.enabled;
                    ImGui::BeginDisabled(!state.supported);
                    if (ImGui::Checkbox("##enabled", &enabled)) {
                        stage.set_enabled(renderer, enabled);
                    }
                    ImGui::EndDisabled();
                    if (!state.supported && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                        ImGui::SetTooltip("%s", std::string{state.reason}.c_str());
                    }
                    ImGui::SameLine();
                }
                ImGui::TextUnformatted(stage.title.data());

                ImGui::TextDisabled("%zu pass%s \xC2\xB7 %s", node.passes.size(), node.passes.size() == 1 ? "" : "es",
                                    queue_name(node.queue));
                ImGui::Text("%.3f ms \xC2\xB7 %u barriers", static_cast<double>(stats.milliseconds), stats.barriers);

                if (stage.draw_settings) {
                    ImGui::SetNextItemOpen(true, ImGuiCond_Once);
                    if (ImGui::TreeNode("Settings")) {
                        ImGui::PushItemWidth(settings_item_width);
                        registry.draw_settings(stage.id, renderer, settings_item_width);
                        ImGui::PopItemWidth();
                        ImGui::TreePop();
                    }
                }

                draw_pins(key);
                ImGui::PopID();
            } else {
                auto const &pass = view.graph.passes[node.passes.front()];
                if (node.culled) {
                    ImGui::BeginDisabled();
                }

                ImGui::TextUnformatted(pass.profile.label.empty() ? pass.name.c_str()
                                                                  : std::string{pass.profile.label}.c_str());
                ImGui::TextDisabled("%s \xC2\xB7 %s%s", pass_type_name(pass.type), queue_name(node.queue),
                                    node.culled ? " \xC2\xB7 culled" : "");
                ImGui::Text("%.3f ms \xC2\xB7 %u barriers", static_cast<double>(stats.milliseconds), stats.barriers);
                draw_pins(key);

                if (node.culled) {
                    ImGui::EndDisabled();
                }
            }

            ImGui::EndGroup();
            ed::EndNode();
            measured_height_[key] = ed::GetNodeSize(node_id_of(key)).y;
        }

        for (auto const &ghost: ghosts_) {
            auto const &stage = stage_list[ghost.stage];

            if (placed_.insert(ghost.key).second) {
                ed::SetNodePosition(node_id_of(ghost.key), ImVec2{ghost.x, ghost.y});
            }

            ed::BeginNode(node_id_of(ghost.key));
            ImGui::BeginGroup();
            ImGui::PushID(stage.id.c_str());

            auto const state = stage.state(renderer);
            if (stage.set_enabled) {
                auto enabled = state.enabled;
                ImGui::BeginDisabled(!state.supported);
                if (ImGui::Checkbox("##enabled", &enabled)) {
                    stage.set_enabled(renderer, enabled);
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
            }
            ImGui::TextUnformatted(stage.title.data());
            ImGui::TextDisabled("Not in the frame graph");

            if (stage.draw_settings) {
                ImGui::SetNextItemOpen(true, ImGuiCond_Once);
                if (ImGui::TreeNode("Settings")) {
                    ImGui::PushItemWidth(settings_item_width);
                    registry.draw_settings(stage.id, renderer, settings_item_width);
                    ImGui::PopItemWidth();
                    ImGui::TreePop();
                }
            }

            ImGui::PopID();
            ImGui::EndGroup();
            ed::EndNode();
            measured_height_[ghost.key] = ed::GetNodeSize(node_id_of(ghost.key)).y;
        }

        for (auto index = std::size_t{0}; index < layout_.edges.size(); ++index) {
            auto const &edge = layout_.edges[index];
            auto const colour = edge.cross_queue ? ImVec4{1.0F, 0.66F, 0.25F, 0.65F} : ImVec4{0.62F, 0.68F, 0.78F, 0.45F};
            ed::Link(ed::LinkId{link_id_tag | index}, output_pin_of(node_keys_[edge.from]),
                     input_pin_of(node_keys_[edge.to]), colour, edge.cross_queue ? 2.5F : 1.5F);
        }

        if (auto const hovered = ed::GetHoveredLink(); hovered) {
            auto const index = static_cast<std::size_t>(hovered.Get() & ~link_id_tag);
            if (index < layout_.edges.size()) {
                auto const &edge = layout_.edges[index];
                auto text = std::string{};
                for (auto const resource: edge.resources) {
                    text += text.empty() ? "" : "\n";
                    text += view.graph.resources[resource].name;
                }
                if (edge.cross_queue) {
                    text += "\n(crosses queues: semaphore wait)";
                }
                ed::Suspend();
                ImGui::SetTooltip("%s", text.c_str());
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
