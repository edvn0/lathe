#pragma once

#include "rendering/environment.hxx"
#include "rendering/file_browser.hxx"
#include "scene/environment.hxx"

namespace gui {
    // The Environment window's contents: source, sky and sun, lighting, fog and the debug views. Edits go straight into
    // `environment`; the return value is whether any did, so the caller can mark the scene dirty. Pressing Browse...
    // opens `browser` and sets `browsing` for the caller to route the picked file to `environment.hdr_source`.
    [[nodiscard]]
    auto draw_environment_panel(SceneEnvironment &environment, EnvironmentSystem &system, FileBrowser &browser,
                                bool &browsing) -> bool;

    // The filters for the browser the panel opens.
    [[nodiscard]]
    auto environment_file_filters() -> std::vector<FileBrowser::Filter>;
} // namespace gui
