#pragma once

#include "rendering/environment.hxx"
#include "rendering/file_browser.hxx"
#include "scene/environment.hxx"

namespace gui {
    [[nodiscard]]
    auto draw_environment_panel(SceneEnvironment &environment, EnvironmentSystem &system, FileBrowser &browser,
                                bool &browsing) -> bool;

    [[nodiscard]]
    auto environment_file_filters() -> std::vector<FileBrowser::Filter>;
}
