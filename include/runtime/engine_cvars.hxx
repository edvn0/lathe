#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "core/cvar.hxx"
#include "gpu/queue_selection.hxx"
#include "runtime/vulkan_bootstrap.hxx"

enum class PresentModeChoice : std::uint8_t {
    immediate,
    mailbox,
    fifo,
    fifo_relaxed,
};

// A WxH window size, e.g. "1920x1080"; nullopt for anything else.
[[nodiscard]] auto parse_window_size(std::string_view text) -> std::optional<std::pair<std::int32_t, std::int32_t>>;

// Every setting the shared startup (run_engine) and the window and swapchain code read. Each is a CVar: --name on the
// command line, LATHE_NAME in the environment. The ones whose default depends on the run (a benchmark turns vsync off,
// say) are read with has_value()/value_or(); the rest with get().
namespace cvars {

    // Packaging and automation.
    extern CVar<std::string> shader_pack;
    extern CVar<std::string> bc7_bench;
    extern CVar<std::uint32_t> cache_max_size;
    extern CVar<bool> clear_cache;
    extern CVar<bool> cache_stats;
    extern CVar<std::string> bake_shaders;
    extern CVar<std::string> record_shaders;
    extern CVar<std::string> record_resources;
    extern CVar<std::string> record_assets;
    extern CVar<std::uint32_t> screenshot_frame;
    extern CVar<bool> screenshot_window;
    extern CVar<std::string> show_panels;
    extern CVar<std::string> inject_keys;
    extern CVar<std::uint32_t> exit_after_frames;
    extern CVar<std::uint32_t> bench_frames;
    extern CVar<bool> bench_pipeline_stats;
    extern CVar<std::uint32_t> bench_warmup;

    // Paths.
    extern CVar<std::string> data_dir;
    extern CVar<std::string> project;

    // Display.
    extern CVar<ScreenType> screen_type;
    extern CVar<std::string> platform;
    extern CVar<std::string> window_size;
    extern CVar<std::string> async_passes;
    extern CVar<AsyncComputeMode> async_compute;
    extern CVar<float> render_scale;
    extern CVar<bool> max_quality;
    extern CVar<std::uint32_t> max_texture_size;
    extern CVar<float> texture_rdo;

    // Presentation.
    extern CVar<bool> vsync;
    extern CVar<PresentModeChoice> present_mode;
    extern CVar<std::uint32_t> swapchain_images;
    extern CVar<bool> hdr;
    extern CVar<float> paper_white;
    extern CVar<float> peak_nits;

    // Diagnostics.
    extern CVar<bool> frame_graph_dump;
    extern CVar<bool> validation;
    extern CVar<std::string> frame_graph_dot;
    extern CVar<bool> rt_poc;
    extern CVar<std::string> rt_poc_view;

    // Game and scene.
    extern CVar<std::string> script;
    extern CVar<bool> player;
    extern CVar<std::filesystem::path> scene;
    extern CVar<std::filesystem::path> save_scene;

}
