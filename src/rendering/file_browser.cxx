#include "rendering/file_browser.hxx"

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <format>
#include <string>
#include <system_error>
#include <utility>

#include "rendering/editor_icons.hxx"

namespace gui {

    namespace {

        [[nodiscard]] auto to_lower(std::string_view text) -> std::string {
            std::string out(text);
            std::ranges::transform(out, out.begin(),
                                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return out;
        }

        [[nodiscard]] auto format_size(std::uintmax_t bytes) -> std::string {
            constexpr std::array<char const *, 4> units{"B", "KiB", "MiB", "GiB"};

            auto value = static_cast<double>(bytes);
            std::size_t unit = 0;
            while (value >= 1024.0 && unit + 1 < units.size()) {
                value /= 1024.0;
                ++unit;
            }

            return unit == 0 ? std::format("{} B", bytes) : std::format("{:.1f} {}", value, units[unit]);
        }

        // Copies `text` into `buffer`, truncating so the terminator always fits.
        template<std::size_t N>
        auto assign_buffer(std::array<char, N> &buffer, std::string_view text) -> void {
            auto const length = std::min(text.size(), N - 1);
            std::copy_n(text.begin(), length, buffer.begin());
            buffer[length] = '\0';
        }

        // Draws `icon` tinted, vertically centred on the current text line. No-op without icons.
        auto draw_icon(EditorIcons const *icons, EditorIcon icon, ImVec4 tint) -> void {
            if (icons == nullptr) {
                return;
            }

            float const size = ImGui::GetTextLineHeight();
            ImGui::ImageWithBg(icons->texture(icon), ImVec2(size, size), ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0),
                               tint);
            ImGui::SameLine();
        }

        constexpr ImVec4 folder_tint{0.95F, 0.80F, 0.45F, 1.0F};
        constexpr ImVec4 file_tint{0.88F, 0.64F, 0.37F, 1.0F};
        constexpr ImVec4 other_file_tint{0.60F, 0.60F, 0.64F, 1.0F};

    } // namespace

    auto path_to_utf8(std::filesystem::path const &path) -> std::string {
        auto const text = path.u8string();
        return {text.begin(), text.end()};
    }

    auto utf8_to_path(std::string_view text) -> std::filesystem::path {
        return std::filesystem::path{std::u8string(text.begin(), text.end())};
    }

    auto FileBrowser::open(std::string title, std::vector<Filter> filters) -> void {
        title_ = std::move(title);
        filters_ = std::move(filters);
        filter_index_ = 0;
        file_name_buffer_[0] = '\0';
        error_.clear();

        std::error_code ec;
        if (directory_.empty() || !std::filesystem::is_directory(directory_, ec)) {
            directory_ = std::filesystem::current_path(ec);
        }

        navigate(directory_);
        open_requested_ = true;
    }

    auto FileBrowser::navigate(std::filesystem::path directory) -> void {
        std::error_code ec;
        auto canonical = std::filesystem::weakly_canonical(directory, ec);
        if (ec) {
            canonical = std::move(directory);
        }

        if (!std::filesystem::is_directory(canonical, ec)) {
            error_ = std::format("'{}' is not a directory", path_to_utf8(canonical));
            assign_buffer(directory_buffer_, path_to_utf8(directory_));
            return;
        }

        directory_ = std::move(canonical);
        error_.clear();
        assign_buffer(directory_buffer_, path_to_utf8(directory_));
        rescan();
    }

    auto FileBrowser::rescan() -> void {
        entries_.clear();

        std::error_code ec;
        std::filesystem::directory_iterator it{directory_, std::filesystem::directory_options::skip_permission_denied,
                                               ec};
        if (ec) {
            error_ = std::format("Could not list '{}': {}", path_to_utf8(directory_), ec.message());
            return;
        }

        for (; it != std::filesystem::directory_iterator{}; it.increment(ec)) {
            if (ec) {
                break;
            }

            auto const &entry = *it;
            auto label = path_to_utf8(entry.path().filename());
            if (label.empty() || label.front() == '.') {
                continue;
            }

            std::error_code entry_ec;
            bool const is_directory = entry.is_directory(entry_ec);
            auto const size = is_directory ? std::uintmax_t{0} : entry.file_size(entry_ec);

            entries_.push_back(Entry{
                    .path = entry.path(),
                    .label = std::move(label),
                    .size = entry_ec ? 0 : size,
                    .is_directory = is_directory,
            });
        }

        std::ranges::sort(entries_, [](Entry const &a, Entry const &b) {
            if (a.is_directory != b.is_directory) {
                return a.is_directory;
            }
            return to_lower(a.label) < to_lower(b.label);
        });
    }

    auto FileBrowser::accepts(std::filesystem::path const &path) const -> bool {
        if (filter_index_ >= filters_.size() || filters_[filter_index_].extensions.empty()) {
            return true;
        }

        auto const extension = to_lower(path_to_utf8(path.extension()));
        return std::ranges::find(filters_[filter_index_].extensions, extension) !=
               filters_[filter_index_].extensions.end();
    }

    auto FileBrowser::draw_places(EditorIcons const *icons) -> void {
        auto place = [&](char const *label, std::filesystem::path const &target) {
            std::error_code ec;
            if (target.empty() || !std::filesystem::is_directory(target, ec)) {
                return;
            }

            ImGui::PushID(label);
            draw_icon(icons, EditorIcon::folder, folder_tint);
            if (ImGui::Selectable(label, directory_ == target)) {
                navigate(target);
            }
            ImGui::PopID();
        };

        std::error_code ec;
        auto const working_directory = std::filesystem::current_path(ec);
        place("Project", working_directory);
        place("Models", working_directory / "assets" / "models");

#if defined(_WIN32)
        char const *home = std::getenv("USERPROFILE");
#else
        char const *home = std::getenv("HOME");
#endif
        if (home != nullptr && home[0] != '\0') {
            place("Home", utf8_to_path(home));
        }

#if defined(_WIN32)
        ImGui::Separator();
        for (char letter = 'A'; letter <= 'Z'; ++letter) {
            auto const root = std::format("{}:\\", letter);
            place(root.c_str(), std::filesystem::path{root});
        }
#else
        place("/", std::filesystem::path{"/"});
#endif
    }

    auto FileBrowser::draw_entries(EditorIcons const *icons) -> std::optional<std::filesystem::path> {
        std::optional<std::filesystem::path> navigate_to;
        std::optional<std::filesystem::path> chosen;

        constexpr ImGuiTableFlags table_flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
                                                ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable;

        if (ImGui::BeginTable("##entries", 2, table_flags, ImVec2(0.0F, 0.0F))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("0000.0 MiB").x);
            ImGui::TableHeadersRow();

            std::string_view const typed_name{file_name_buffer_.data()};

            for (auto const &entry: entries_) {
                if (!entry.is_directory && !accepts(entry.path)) {
                    continue;
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::PushID(entry.label.c_str());

                bool const is_selected = typed_name == entry.label;
                if (ImGui::Selectable("##row", is_selected,
                                      ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick |
                                              ImGuiSelectableFlags_AllowOverlap)) {
                    assign_buffer(file_name_buffer_, entry.label);

                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                        if (entry.is_directory) {
                            navigate_to = entry.path;
                        } else {
                            chosen = entry.path;
                        }
                    }
                }

                ImGui::SameLine();
                draw_icon(icons, entry.is_directory ? EditorIcon::folder : EditorIcon::mesh,
                          entry.is_directory ? folder_tint : (accepts(entry.path) ? file_tint : other_file_tint));
                ImGui::TextUnformatted(entry.label.c_str());

                ImGui::TableNextColumn();
                if (!entry.is_directory) {
                    ImGui::TextDisabled("%s", format_size(entry.size).c_str());
                }

                ImGui::PopID();
            }

            ImGui::EndTable();
        }

        // Applied after the loop, since navigating rebuilds entries_.
        if (navigate_to) {
            navigate(std::move(*navigate_to));
            file_name_buffer_[0] = '\0';
        }

        return chosen;
    }

    auto FileBrowser::draw(EditorIcons const *icons) -> std::optional<std::filesystem::path> {
        constexpr char const *popup_id = "###file_browser";

        if (open_requested_) {
            ImGui::OpenPopup(popup_id);
            open_requested_ = false;
            visible_ = true;
        }

        if (!visible_) {
            return std::nullopt;
        }

        auto const *viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5F, 0.5F));
        ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x * 0.6F, viewport->WorkSize.y * 0.6F), ImGuiCond_Appearing);

        auto const label = std::format("{}{}", title_, popup_id);
        bool keep_open = true;
        if (!ImGui::BeginPopupModal(label.c_str(), &keep_open)) {
            visible_ = false;
            return std::nullopt;
        }

        std::optional<std::filesystem::path> chosen;
        auto const &style = ImGui::GetStyle();

        // Directory bar: up, then an editable path (Enter navigates).
        if (ImGui::ArrowButton("##up", ImGuiDir_Up) && directory_.has_parent_path() &&
            directory_.parent_path() != directory_) {
            navigate(directory_.parent_path());
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Parent directory");
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1.0F);
        if (ImGui::InputText("##directory", directory_buffer_.data(), directory_buffer_.size(),
                             ImGuiInputTextFlags_EnterReturnsTrue)) {
            navigate(utf8_to_path(directory_buffer_.data()));
        }

        if (!error_.empty()) {
            ImGui::TextColored(ImVec4(1.0F, 0.45F, 0.40F, 1.0F), "%s", error_.c_str());
        }

        float const footer_height = ImGui::GetFrameHeightWithSpacing() * 2.0F + style.ItemSpacing.y;
        float const places_width = ImGui::GetFontSize() * 10.0F;

        if (ImGui::BeginChild("##places", ImVec2(places_width, -footer_height), ImGuiChildFlags_Borders)) {
            draw_places(icons);
        }
        ImGui::EndChild();

        ImGui::SameLine();

        if (ImGui::BeginChild("##listing", ImVec2(0.0F, -footer_height))) {
            chosen = draw_entries(icons);
        }
        ImGui::EndChild();

        // Footer: file name and filter, then the buttons.
        bool confirm = false;

        float const filter_width = ImGui::GetFontSize() * 16.0F;
        ImGui::SetNextItemWidth(-(filter_width + style.ItemSpacing.x));
        if (ImGui::InputTextWithHint("##file_name", "File name", file_name_buffer_.data(), file_name_buffer_.size(),
                                     ImGuiInputTextFlags_EnterReturnsTrue)) {
            confirm = true;
        }

        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1.0F);
        auto const *current_filter = filter_index_ < filters_.size() ? filters_[filter_index_].label.c_str() : "";
        if (ImGui::BeginCombo("##filter", current_filter)) {
            for (std::size_t i = 0; i < filters_.size(); ++i) {
                if (ImGui::Selectable(filters_[i].label.c_str(), i == filter_index_)) {
                    filter_index_ = i;
                }
            }
            ImGui::EndCombo();
        }

        float const button_width = ImGui::GetFontSize() * 6.0F;
        ImGui::SetCursorPosX(ImGui::GetContentRegionMax().x - button_width * 2.0F - style.ItemSpacing.x);

        bool const has_name = file_name_buffer_[0] != '\0';
        ImGui::BeginDisabled(!has_name);
        confirm |= ImGui::Button("Open", ImVec2(button_width, 0.0F));
        ImGui::EndDisabled();

        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(button_width, 0.0F)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            keep_open = false;
        }

        if (confirm && has_name && !chosen) {
            auto const typed = utf8_to_path(file_name_buffer_.data());
            auto const target = typed.is_absolute() ? typed : directory_ / typed;

            std::error_code ec;
            if (std::filesystem::is_directory(target, ec)) {
                navigate(target);
                file_name_buffer_[0] = '\0';
            } else if (std::filesystem::is_regular_file(target, ec)) {
                chosen = target;
            } else {
                error_ = std::format("'{}' does not exist", path_to_utf8(target));
            }
        }

        if (chosen || !keep_open) {
            visible_ = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
        return chosen;
    }

} // namespace gui
