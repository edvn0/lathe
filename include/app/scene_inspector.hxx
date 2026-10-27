#pragma once

#include <array>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>

#include "rendering/file_browser.hxx"
#include "serialisation/scene_inspect.hxx"

namespace gui {

    class EditorIcons;

    // A panel that opens a scene file and shows what is in it without loading it into the editor: container version
    // and chunks, the SCEN sections with their versions (and which are older or newer than this build), the shared
    // strings, and whether the scene decodes.
    class SceneInspector {
    public:
        // `current_scene` is the file open in the editor, offered as a one-click target; may be empty.
        auto draw(EditorIcons const *icons, std::filesystem::path const &current_scene, bool &open) -> void;

        auto inspect(std::filesystem::path const &path) -> void;

    private:
        auto draw_info(SceneFileInfo const &info) -> void;
        auto draw_scene(InspectedScene const &scene) -> void;

        FileBrowser browser_;
        std::array<char, 512> path_buffer_{};
        std::string string_filter_;

        std::optional<std::expected<SceneFileInfo, LbfError>> result_;
    };

} // namespace gui
