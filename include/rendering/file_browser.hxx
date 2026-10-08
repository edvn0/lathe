#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gui {

    class EditorIcons;

    [[nodiscard]] auto path_to_utf8(std::filesystem::path const &path) -> std::string;
    [[nodiscard]] auto utf8_to_path(std::string_view text) -> std::filesystem::path;

    class FileBrowser {
    public:
        struct Filter {
            std::string label;
            std::vector<std::string> extensions;
        };

        auto open(std::string title, std::vector<Filter> filters) -> void;

        [[nodiscard]] auto is_open() const noexcept -> bool { return visible_; }

        [[nodiscard]] auto draw(EditorIcons const *icons) -> std::optional<std::filesystem::path>;

    private:
        struct Entry {
            std::filesystem::path path;
            std::string label;
            std::uintmax_t size = 0;
            bool is_directory = false;
        };

        auto navigate(std::filesystem::path directory) -> void;
        auto rescan() -> void;

        [[nodiscard]] auto accepts(std::filesystem::path const &path) const -> bool;

        auto draw_places(EditorIcons const *icons) -> void;
        [[nodiscard]] auto draw_entries(EditorIcons const *icons) -> std::optional<std::filesystem::path>;

        std::string title_;
        std::vector<Filter> filters_;
        std::size_t filter_index_ = 0;

        std::filesystem::path directory_;
        std::vector<Entry> entries_;
        std::string error_;

        std::array<char, 1024> directory_buffer_{};
        std::array<char, 256> file_name_buffer_{};

        bool open_requested_ = false;
        bool visible_ = false;
    };

}
