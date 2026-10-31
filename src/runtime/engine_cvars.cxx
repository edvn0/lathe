#include "runtime/engine_cvars.hxx"

#include <array>
#include <cstdio>

auto parse_window_size(std::string_view text) -> std::optional<std::pair<std::int32_t, std::int32_t>> {
    std::int32_t width = 0;
    std::int32_t height = 0;
    char trailing = '\0';

    // sscanf wants a terminated string; a window size is a few characters.
    auto const owned = std::string{text};

    if (std::sscanf(owned.c_str(), "%dx%d%c", &width, &height, &trailing) != 2 || width <= 0 || height <= 0) {
        return std::nullopt;
    }

    return std::pair{width, height};
}

namespace {

    constexpr std::array<std::pair<std::string_view, ScreenType>, 4> screen_type_choices{{
            {"windowed", ScreenType::windowed},
            {"fullscreen", ScreenType::fullscreen},
            {"borderless", ScreenType::borderless},
            {"headless", ScreenType::headless},
    }};

    constexpr std::array<std::pair<std::string_view, AsyncComputeMode>, 3> async_compute_choices{{
            {"auto", AsyncComputeMode::automatic},
            {"off", AsyncComputeMode::off},
            {"same_family", AsyncComputeMode::same_family},
    }};

    constexpr std::array<std::pair<std::string_view, PresentModeChoice>, 4> present_mode_choices{{
            {"immediate", PresentModeChoice::immediate},
            {"mailbox", PresentModeChoice::mailbox},
            {"fifo", PresentModeChoice::fifo},
            {"fifo_relaxed", PresentModeChoice::fifo_relaxed},
    }};

    auto check_platform(std::string_view text) -> std::expected<void, std::string> {
        if (text == "x11" || text == "auto") {
            return {};
        }

        return std::unexpected(std::format("'{}' (expected x11 or auto)", text));
    }

    auto check_window_size(std::string_view text) -> std::expected<void, std::string> {
        if (parse_window_size(text)) {
            return {};
        }

        return std::unexpected(std::format("'{}' is not WIDTHxHEIGHT", text));
    }

}

namespace cvars {

    CVar<std::string> shader_pack{{
            .name = "shader_pack",
            .group = "Packaging",
            .help = "Precompiled shaders to use (default: shaders.lsp in the data directory, else a cache under "
                    "$XDG_CACHE_HOME/lathe that is filled automatically)",
            .hint = "FILE.lsp",
    }};
    CVar<bool> gpu_texture_encoder{{
            .name = "gpu_texture_encoder",
            .group = "Packaging",
            .help = "Compress textures to BC7 on the GPU (a spare compute queue), which is far faster than the CPU encoder; "
                    "off: always on the CPU. Normal maps are always compressed on the CPU",
            .default_value = true,
    }};
    CVar<std::string> gpu_texture_bench{{
            .name = "gpu_texture_bench",
            .group = "Packaging",
            .help = "Compress an 8-bit RGBA image to BC7 on the GPU and with the CPU path, print times and quality, then "
                    "exit (prototype)",
            .hint = "IMAGE",
            .cli_only = true,
    }};
    CVar<std::uint32_t> cache_max_size{{
            .name = "cache_max_size",
            .group = "Packaging",
            .help = "Size budget of the on-disk cache (textures, shader binaries and packs) in MiB; the least recently "
                    "used entries beyond it are deleted at start-up (0: no limit)",
            .hint = "MIB",
            .default_value = 8192,
    }};
    CVar<bool> clear_cache{{
            .name = "clear_cache",
            .group = "Packaging",
            .help = "Delete everything in the on-disk cache, and the caches of older versions, then exit",
            .cli_only = true,
    }};
    CVar<bool> cache_stats{{
            .name = "cache_stats",
            .group = "Packaging",
            .help = "Print what the on-disk cache holds, then exit",
            .cli_only = true,
    }};
    CVar<std::string> bake_shaders{{
            .name = "bake_shaders",
            .group = "Packaging",
            .help = "Compile every shader under assets/shaders (plus assets/shaders/variants.txt) into a shader pack, "
                    "then exit; needs no GPU or window",
            .hint = "FILE.lsp",
            .cli_only = true,
    }};
    CVar<std::string> record_shaders{{
            .name = "record_shaders",
            .group = "Packaging",
            .help = "Write every shader this run compiles to a shader pack, then exit",
            .hint = "FILE.lsp",
            .cli_only = true,
    }};
    CVar<std::string> record_resources{{
            .name = "record_resources",
            .group = "Packaging",
            .help = "Write the editor font and icons this run loaded to a resource pack, then exit",
            .hint = "FILE.lbf",
            .cli_only = true,
    }};
    CVar<std::string> record_assets{{
            .name = "record_assets",
            .group = "Packaging",
            .help = "Write the data files this run opened to a list, then exit",
            .hint = "FILE.txt",
            .cli_only = true,
    }};
    CVar<std::uint32_t> screenshot_frame{{
            .name = "screenshot_frame",
            .group = "Packaging",
            .help = "Save a screenshot on frame N (see the screenshots directory)",
            .hint = "N",
    }};
    CVar<bool> screenshot_window{{
            .name = "screenshot_window",
            .group = "Packaging",
            .help = "Screenshots capture the whole editor window, not just the viewport",
    }};
    CVar<std::string> show_panels{{
            .name = "show_panels",
            .group = "Packaging",
            .help = "Open these editor panels at startup, e.g. frame_timings,scene_stats (names as in the Window menu)",
            .hint = "NAME,...",
    }};
    CVar<std::string> inject_keys{{
            .name = "inject_keys",
            .group = "Packaging",
            .help = "Press GLFW key codes on given frames, e.g. 120:257 presses Enter on frame 120; prefix ctrl+, "
                    "shift+ or alt+ for modifiers, e.g. 120:ctrl+82 is Ctrl+R",
            .hint = "FRAME:KEY,...",
    }};
    CVar<std::uint32_t> exit_after_frames{{
            .name = "exit_after_frames",
            .group = "Packaging",
            .help = "Exit after N frames (0: run until closed)",
            .hint = "N",
    }};
    CVar<std::uint32_t> bench_frames{{
            .name = "bench_frames",
            .group = "Packaging",
            .help = "Average per-pass GPU time over N frames after warm-up, print it, then exit",
            .hint = "N",
    }};
    CVar<bool> bench_pipeline_stats{{
            .name = "bench_pipeline_stats",
            .group = "Packaging",
            .help = "Keep the task/mesh/fragment invocation query on while benchmarking (it is slow: measures the "
                    "old behaviour)",
    }};
    CVar<std::uint32_t> bench_warmup{{
            .name = "bench_warmup",
            .group = "Packaging",
            .help = "Frames to run before --bench-frames starts sampling (default 120)",
            .hint = "N",
            .default_value = 120,
    }};

    CVar<std::string> data_dir{{
            .name = "data_dir",
            .group = "Paths",
            .help = "Game data directory (default: an installed game's data/, else the working directory)",
            .hint = "DIR",
    }};
    CVar<std::string> project{{
            .name = "project",
            .group = "Paths",
            .help = "A lathe.project or the directory holding one (default: the nearest one at or above the working "
                    "directory)",
            .hint = "PATH",
    }};

    CVar<ScreenType> screen_type{{
            .name = "screen_type",
            .group = "Display",
            .help = "Window mode (default fullscreen)",
            .default_value = ScreenType::fullscreen,
            .choices = screen_type_choices,
    }};
    CVar<std::string> platform{{
            .name = "platform",
            .group = "Display",
            .help = "Window system on Linux: x11 runs through XWayland, where ImGui windows can be dragged out of the "
                    "main one; auto (default) takes what GLFW finds",
            .hint = "x11|auto",
            .default_value = "auto",
            .validator = check_platform,
    }};
    CVar<std::string> window_size{{
            .name = "window_size",
            .group = "Display",
            .help = "Size of a windowed or headless window, e.g. 1920x1080 for full-HD screenshots (default 1280x720)",
            .hint = "WxH",
            .validator = check_window_size,
    }};
    CVar<std::string> async_passes{{
            .name = "async_passes",
            .group = "Display",
            .help = "Passes that run on the async compute queue when it exists: light, occlusion, gtao (comma "
                    "separated, always on), none, or auto (default: GTAO only while frames are heavy enough to "
                    "overlap it)",
            .hint = "LIST",
    }};
    CVar<AsyncComputeMode> async_compute{{
            .name = "async_compute",
            .group = "Display",
            .help = "Run compute passes on a separate queue family (auto), on the graphics queue (off), or on a "
                    "second queue of the graphics family (same_family)",
            .default_value = AsyncComputeMode::automatic,
            .choices = async_compute_choices,
    }};
    CVar<float> render_scale{{
            .name = "render_scale",
            .group = "Display",
            .help = "Draw the scene at the viewport size times FACTOR (0.2 to 3.5, default 1); the editor has a "
                    "slider for it",
            .hint = "FACTOR",
            .default_value = 1.0F,
    }};
    CVar<bool> max_quality{{
            .name = "max_quality",
            .group = "Display",
            .help = "Turn every rendering feature on and up: 4x MSAA, fog, up to 64 shadowed local lights, heavier "
                    "AO, all culling and clustered lighting",
    }};
    CVar<std::uint32_t> max_texture_size{{
            .name = "max_texture_size",
            .group = "Display",
            .help = "Largest side of any loaded texture; bigger ones lose their top mips (0 = no limit)",
            .hint = "TEXELS",
    }};
    CVar<float> texture_rdo{{
            .name = "texture_rdo",
            .group = "Display",
            .help = "Smaller texture files at some quality cost: UASTC RDO strength, about 0.25 to 10 (0 = off). "
                    "Textures are re-encoded and cached separately",
            .hint = "LAMBDA",
    }};

    CVar<bool> vsync{{
            .name = "vsync",
            .group = "Presentation",
            .help = "Cap the frame rate at the refresh rate (off by default while benchmarking)",
            .toggle = true,
            .default_label = "auto",
    }};
    CVar<PresentModeChoice> present_mode{{
            .name = "present_mode",
            .group = "Presentation",
            .help = "Present mode outright (benchmarks prefer immediate: MAILBOX can still pace acquisition to the "
                    "refresh rate)",
            .choices = present_mode_choices,
            .default_label = "auto",
    }};
    CVar<std::uint32_t> swapchain_images{{
            .name = "swapchain_images",
            .group = "Presentation",
            .help = "Swapchain images to ask for, 2 to 8",
            .hint = "N",
            .min = 2,
            .max = 8,
            .default_label = "auto",
    }};
    CVar<bool> hdr{{
            .name = "hdr",
            .group = "Presentation",
            .help = "Output linear scRGB on an HDR display (falls back to sRGB when the surface offers none; on by "
                    "default except in benchmarks and tests)",
            .toggle = true,
            .default_label = "auto",
    }};
    CVar<float> paper_white{{
            .name = "paper_white",
            .group = "Presentation",
            .help = "HDR paper white, the brightness of an sRGB 1.0 (default 203)",
            .hint = "NITS",
            .default_value = 203.0F,
            .min = 80.0F,
            .max = 1000.0F,
    }};
    CVar<float> peak_nits{{
            .name = "peak_nits",
            .group = "Presentation",
            .help = "HDR peak brightness the tonemapper rolls off to (default 1000)",
            .hint = "NITS",
            .default_value = 1000.0F,
            .min = 80.0F,
            .max = 10000.0F,
    }};

    CVar<bool> frame_graph_dump{{
            .name = "frame_graph_dump",
            .group = "Diagnostics",
            .help = "Log the compiled frame graph whenever it changes",
    }};
    CVar<bool> validation{{
            .name = "validation",
            .group = "Diagnostics",
            .help = "Load the Vulkan validation layer (on by default in Debug builds, off in Release)",
            .toggle = true,
            .default_label = "auto",
    }};
    CVar<std::string> frame_graph_dot{{
            .name = "frame_graph_dot",
            .group = "Diagnostics",
            .help = "Write the compiled frame graph as Graphviz whenever it changes (render with `dot -Tsvg`).",
            .hint = "FILE.dot",
    }};
    CVar<bool> rt_poc{{
            .name = "rt_poc",
            .group = "Diagnostics",
            .help = "Trace a sphere into the scene with VK_KHR_ray_query (docs/ray-tracing.md)",
    }};
    CVar<std::string> rt_poc_view{{
            .name = "rt_poc_view",
            .group = "Diagnostics",
            .help = "Debug view of --rt-poc: beauty, normals, distance, shadow-mask, triangle, barycentrics, "
                    "depth-test",
            .hint = "NAME",
            .default_value = "beauty",
    }};

    CVar<std::string> script{{
            .name = "script",
            .group = "Game",
            .help = "Entry script of a Lua game (--game lua), relative to the data directory",
            .hint = "FILE.lua",
    }};
    CVar<bool> player{{
            .name = "player",
            .group = "Game",
            .help = "Run as an installed game would: fullscreen play, no editor",
    }};

    CVar<std::filesystem::path> scene{{
            .name = "scene",
            .group = "Scene",
            .help = "Open a saved scene in place of the game's",
            .hint = "FILE.lbf",
            .cli_only = true,
    }};
    CVar<std::filesystem::path> save_scene{{
            .name = "save_scene",
            .group = "Scene",
            .help = "Cook the (opened) scene into a self-contained .lbf",
            .hint = "FILE.lbf",
            .cli_only = true,
    }};

}
