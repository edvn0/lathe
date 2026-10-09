#pragma once

#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/fly_string.hxx"
#include "rendering/cluster_grid.hxx"

class Renderer;

namespace stages {

    struct State {
        bool enabled = true;
        bool supported = true;
        std::string_view reason;
    };

    struct UiState {
        int hiz_debug_mip = 0;
        std::optional<ClusterGridSettings> refused_cluster_grid;
    };

    struct Context {
        Renderer &renderer;
        UiState &ui;
        // Previews fill the available width unless this is positive. Needed where the available width follows the
        // content, as in a node.
        float max_preview_width = 0.0F;
    };

    // A user-facing renderer stage. Unlike a frame graph pass it exists whether or not the graph currently contains
    // it, so a disabled stage can still be shown and switched back on. A stage may own several passes.
    struct Stage {
        FlyString id;
        std::string_view title;
        std::vector<FlyString> passes;
        std::function<State(Renderer const &)> state;
        std::function<void(Renderer &, bool)> set_enabled;
        std::function<void(Context &)> draw_settings;
    };

    class Registry {
    public:
        Registry();

        [[nodiscard]] auto stages() const noexcept -> std::span<Stage const> { return stages_; }
        [[nodiscard]] auto find(FlyString id) const -> Stage const *;
        [[nodiscard]] auto stage_of_pass(FlyString pass) const -> Stage const *;

        auto draw_settings(FlyString id, Renderer &renderer, float max_preview_width = 0.0F) -> void;

    private:
        auto add(Stage stage) -> void;

        std::vector<Stage> stages_;
        std::unordered_map<FlyString, std::size_t> by_id_;
        std::unordered_map<FlyString, std::size_t> by_pass_;
        UiState ui_;
    };

}
