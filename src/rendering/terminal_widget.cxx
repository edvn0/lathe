#include "rendering/terminal_widget.hxx"

#include <imgui.h>

#include <cstddef>
#include <string>
#include <utility>

namespace gui {

    TerminalWidget::MessageRing::MessageRing(std::size_t capacity) : capacity_{capacity} {
        //
        // Reserve raw storage once, but don't construct 10,000 strings
        // up front.
        //
        // This gives us one allocation for the ConsoleMessage array while
        // only constructing elements as messages actually arrive.
        //
        storage_.reserve(capacity_);
    }

    auto TerminalWidget::MessageRing::push(logger::ConsoleMessage message) -> void {
        if (capacity_ == 0) {
            return;
        }

        if (storage_.size() < capacity_) {
            storage_.push_back(std::move(message));

            return;
        }

        //
        // Full ring: replace the oldest message.
        //
        // Moving the ConsoleMessage transfers ownership of the message
        // string without copying its payload.
        //
        storage_[write_index_] = std::move(message);

        ++write_index_;

        if (write_index_ == capacity_) {
            write_index_ = 0;
        }
    }

    auto TerminalWidget::MessageRing::clear() -> void {
        //
        // Destroy all retained strings so their payload allocations are
        // released, but retain the vector's raw ConsoleMessage allocation
        // for future use.
        //
        storage_.clear();
        write_index_ = 0;
    }

    auto TerminalWidget::MessageRing::size() const noexcept -> std::size_t { return storage_.size(); }

    auto TerminalWidget::MessageRing::empty() const noexcept -> bool { return storage_.empty(); }

    auto TerminalWidget::MessageRing::operator[](std::size_t index) const noexcept -> logger::ConsoleMessage const & {
        //
        // Before reaching capacity the vector is already chronological.
        //
        if (storage_.size() < capacity_) {
            return storage_[index];
        }

        //
        // Once full, write_index_ always points at the oldest element.
        //
        auto physical_index = write_index_ + index;

        if (physical_index >= capacity_) {
            physical_index -= capacity_;
        }

        return storage_[physical_index];
    }

    constexpr auto TerminalWidget::level_index(logger::Level level) noexcept -> std::size_t {
        return static_cast<std::size_t>(level);
    }

    constexpr auto TerminalWidget::level_prefix(logger::Level level) noexcept -> char const * {
        switch (level) {
            case logger::Level::trace:
                return "[T]";

            case logger::Level::debug:
                return "[D]";

            case logger::Level::info:
                return "[I]";

            case logger::Level::warn:
                return "[W]";

            case logger::Level::error:
                return "[E]";

            case logger::Level::fatal:
                return "[F]";
        }

        return "[?]";
    }

    constexpr auto TerminalWidget::level_colour(logger::Level level) noexcept -> ImVec4 {
        switch (level) {
            case logger::Level::trace:
                return ImVec4{
                        0.60F,
                        0.60F,
                        0.60F,
                        1.00F,
                };

            case logger::Level::debug:
                return ImVec4{
                        0.65F,
                        0.75F,
                        0.95F,
                        1.00F,
                };

            case logger::Level::info:
                return ImVec4{
                        0.85F,
                        0.85F,
                        0.85F,
                        1.00F,
                };

            case logger::Level::warn:
                return ImVec4{
                        1.00F,
                        0.80F,
                        0.25F,
                        1.00F,
                };

            case logger::Level::error:
                return ImVec4{
                        1.00F,
                        0.35F,
                        0.35F,
                        1.00F,
                };

            case logger::Level::fatal:
                return ImVec4{
                        1.00F,
                        0.20F,
                        0.70F,
                        1.00F,
                };
        }

        return ImVec4{
                1.00F,
                1.00F,
                1.00F,
                1.00F,
        };
    }

    TerminalWidget::TerminalWidget(std::size_t max_messages) : history_{max_messages} {
        //
        // Most frames will contain relatively few log records. This avoids
        // the first couple of tiny reallocations without reserving anything
        // significant compared with the retained history.
        //
        pending_.reserve(64);
        visible_indices_.reserve(max_messages);
    }

    auto TerminalWidget::clear() -> void {
        history_.clear();
        visible_indices_.clear();
        visible_has_multiline_messages_ = false;
    }

    auto TerminalWidget::drain_new_messages() -> bool {
        logger::Logger::the().drain_console(pending_);

        if (pending_.empty()) {
            return false;
        }

        for (auto &message: pending_) {
            history_.push(std::move(message));
        }

        //
        // Keep the allocation around. On the next drain this vector is
        // swapped back into the sink and reused by logging threads.
        //
        pending_.clear();

        return true;
    }

    auto TerminalWidget::draw_level_checkbox(char const *label, logger::Level level) -> bool {
        return ImGui::Checkbox(label, &enabled_levels_[level_index(level)]);
    }

    auto TerminalWidget::draw_toolbar() -> bool {
        bool filters_changed = false;

        if (ImGui::Button("Clear")) {
            clear();
            filters_changed = true;
        }

        ImGui::SameLine();

        filters_changed |= filter_.Draw("Filter", 260.0F);

        ImGui::SameLine();

        ImGui::Checkbox("Auto-scroll", &auto_scroll_);

        ImGui::SameLine();

        ImGui::Checkbox("Wrap", &wrap_text_);

        ImGui::Separator();

        ImGui::TextUnformatted("Levels:");

        ImGui::SameLine();

        filters_changed |= draw_level_checkbox("Trace", logger::Level::trace);

        ImGui::SameLine();

        filters_changed |= draw_level_checkbox("Debug", logger::Level::debug);

        ImGui::SameLine();

        filters_changed |= draw_level_checkbox("Info", logger::Level::info);

        ImGui::SameLine();

        filters_changed |= draw_level_checkbox("Warn", logger::Level::warn);

        ImGui::SameLine();

        filters_changed |= draw_level_checkbox("Error", logger::Level::error);

        ImGui::SameLine();

        filters_changed |= draw_level_checkbox("Fatal", logger::Level::fatal);

        return filters_changed;
    }

    auto TerminalWidget::rebuild_visible_indices() -> void {
        visible_indices_.clear();

        visible_has_multiline_messages_ = false;

        for (std::size_t i = 0; i < history_.size(); ++i) {
            auto const &entry = history_[i];

            if (!enabled_levels_[level_index(entry.level)]) {
                continue;
            }

            if (!filter_.PassFilter(entry.message.c_str())) {
                continue;
            }

            visible_indices_.push_back(i);

            if (!visible_has_multiline_messages_ && entry.message.find('\n') != std::string::npos) {
                visible_has_multiline_messages_ = true;
            }
        }
    }

    auto TerminalWidget::copy_message_with_level(logger::ConsoleMessage const &message) -> void {
        auto text = std::string{
                level_prefix(message.level),
        };

        text += ' ';
        text += message.message;

        ImGui::SetClipboardText(text.c_str());
    }

    auto TerminalWidget::draw_message(std::size_t message_index) -> void {
        auto const &entry = history_[message_index];

        ImGui::PushID(static_cast<int>(message_index));

        ImGui::PushStyleColor(ImGuiCol_Text, level_colour(entry.level));

        ImGui::TextUnformatted(level_prefix(entry.level));

        ImGui::PopStyleColor();

        ImGui::SameLine(0.0F, ImGui::GetStyle().ItemSpacing.x);

        if (wrap_text_) {
            ImGui::PushTextWrapPos(0.0F);
        }

        ImGui::TextUnformatted(entry.message.data(), entry.message.data() + entry.message.size());

        if (wrap_text_) {
            ImGui::PopTextWrapPos();
        }

        if (ImGui::BeginPopupContextItem("##terminal_message_context")) {
            if (ImGui::MenuItem("Copy message")) {
                ImGui::SetClipboardText(entry.message.c_str());
            }

            if (ImGui::MenuItem("Copy with level")) {
                copy_message_with_level(entry);
            }

            ImGui::EndPopup();
        }

        ImGui::PopID();
    }

    auto TerminalWidget::draw_all_messages() -> void {
        for (auto const message_index: visible_indices_) {
            draw_message(message_index);
        }
    }

    auto TerminalWidget::draw_clipped_messages() -> void {
        ImGuiListClipper clipper;

        clipper.Begin(static_cast<int>(visible_indices_.size()));

        while (clipper.Step()) {
            for (auto row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                auto const message_index = visible_indices_[static_cast<std::size_t>(row)];

                draw_message(message_index);
            }
        }
    }

    auto TerminalWidget::draw() -> void {
        auto const had_new_messages = drain_new_messages();

        auto const filters_changed = draw_toolbar();

        visibility_dirty_ |= had_new_messages || filters_changed;

        if (visibility_dirty_) {
            rebuild_visible_indices();
            visibility_dirty_ = false;
        }

        ImGui::TextDisabled("%zu / %zu messages", visible_indices_.size(), history_.size());

        ImGui::Separator();

        auto const window_flags = wrap_text_ ? ImGuiWindowFlags_None : ImGuiWindowFlags_HorizontalScrollbar;

        if (ImGui::BeginChild("##terminal_scrolling", ImVec2{0.0F, 0.0F}, ImGuiChildFlags_None, window_flags)) {
            auto const was_at_bottom = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 2.0F;

            if (wrap_text_ || visible_has_multiline_messages_) {
                draw_all_messages();
            } else {
                draw_clipped_messages();
            }

            if (auto_scroll_ && had_new_messages && was_at_bottom && !visible_indices_.empty()) {
                ImGui::SetScrollHereY(1.0F);
            }
        }

        ImGui::EndChild();
    }

} // namespace gui
