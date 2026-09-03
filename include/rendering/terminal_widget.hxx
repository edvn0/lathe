#pragma once

#include "core/logger.hxx"

#include <imgui.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace gui {

    struct TerminalWidget {
    public:
        explicit TerminalWidget(std::size_t max_messages = 10'000);

        auto draw() -> void;

        auto clear() -> void;

    private:
        class MessageRing {
        public:
            explicit MessageRing(std::size_t capacity);

            auto push(logger::ConsoleMessage message) -> void;

            auto clear() -> void;

            [[nodiscard]]
            auto size() const noexcept -> std::size_t;

            [[nodiscard]]
            auto empty() const noexcept -> bool;

            [[nodiscard]]
            auto operator[](std::size_t index) const noexcept -> logger::ConsoleMessage const &;

        private:
            std::size_t capacity_{};
            std::size_t write_index_{};

            std::vector<logger::ConsoleMessage> storage_;
        };

        static constexpr auto level_count = static_cast<std::size_t>(logger::Level::fatal) + 1;

        [[nodiscard]]
        static constexpr auto level_index(logger::Level level) noexcept -> std::size_t;

        [[nodiscard]]
        static constexpr auto level_prefix(logger::Level level) noexcept -> char const *;

        [[nodiscard]]
        static constexpr auto level_colour(logger::Level level) noexcept -> ImVec4;

        auto drain_new_messages() -> bool;

        auto draw_toolbar() -> bool;

        auto draw_level_checkbox(char const *label, logger::Level level) -> bool;

        auto rebuild_visible_indices() -> void;

        auto draw_all_messages() -> void;

        auto draw_clipped_messages() -> void;

        auto draw_message(std::size_t message_index) -> void;

        static auto copy_message_with_level(logger::ConsoleMessage const &message) -> void;

        MessageRing history_;

        //
        // This vector participates in the swap-based producer/consumer
        // exchange with TerminalSink.
        //
        // Its allocation is recycled rather than recreated every frame.
        //
        std::vector<logger::ConsoleMessage> pending_;

        //
        // Logical indices into history_, not physical indices into its
        // underlying ring storage.
        //
        std::vector<std::size_t> visible_indices_;

        ImGuiTextFilter filter_;

        std::array<bool, level_count> enabled_levels_{
                true, // trace
                true, // debug
                true, // info
                true, // warn
                true, // error
                true, // fatal
        };

        bool auto_scroll_{true};
        bool wrap_text_{true};
        bool visibility_dirty_{true};

        //
        // ImGuiListClipper assumes uniform line height. Even with wrapping
        // disabled, an individual log record could contain '\n'.
        //
        bool visible_has_multiline_messages_{false};
    };

} // namespace gui
