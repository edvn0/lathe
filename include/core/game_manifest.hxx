#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

// The probe global illumination (docs/probe-gi.md): on by default wherever the device has VK_KHR_ray_query, tuned in the
// `[gi]` table of lathe.project / game.toml, with the editor's Project Settings page and the frame graph's Probe GI stage
// for changing it while the editor runs.
struct GiSettings {
    bool enabled = true;
    // Rays per probe per frame, a multiple of 64 up to 256.
    std::uint32_t rays_per_probe = 64;
    // Metres between the probes of the near cascade; the coarse cascade is four times that.
    float probe_spacing = 1.0F;
    // Scales the diffuse light the probes give.
    float intensity = 1.0F;
    // How strongly small emissive pieces (bulbs, lantern glass) light their surroundings, like a point light's intensity.
    float emitter_intensity = 200.0F;
    // The second, coarser cascade that reaches four times as far.
    bool far_cascade = true;
    // The far cascade updates every this many frames; 0 follows the performance tier.
    std::uint32_t far_update_interval = 0;
    // The point lights a model gets from its emissive meshes when it loads: "auto" (none while the probes light them
    // themselves), "on" or "off".
    std::string synthesize_lights = "auto";

    auto operator==(GiSettings const &) const -> bool = default;
};

// The `[render]` table: renderer features that are on wherever the device can do them.
struct RenderSettings {
    // The sun's shadow is ray traced against the scene (docs/ray-traced-shadows.md) instead of coming from the cascaded
    // shadow map, where the device has VK_KHR_ray_query.
    bool ray_traced_shadows = true;
    // The GPU class that picks the defaults of the quality settings left on "auto": "auto" (detected from the device type
    // and memory), "low", "medium" or "high".
    std::string performance_tier = "auto";
    // The resolution the ray traced shadow mask is traced at: "full", "half" or "quarter" of the screen, or "auto" (follows
    // the performance tier: full on high, half on medium, quarter on low).
    std::string ray_traced_shadow_resolution = "auto";

    auto operator==(RenderSettings const &) const -> bool = default;
};

// The `[diagnostics]` table.
struct DiagnosticsSettings {
    // GPU crash dumps through the Nsight Aftermath SDK (docs/gpu-crash-dumps.md), where the build has it: "auto" (the cheap
    // collection, and everything after a device-lost restart), "standard", "full" or "off".
    std::string gpu_crash_dumps = "auto";

    auto operator==(DiagnosticsSettings const &) const -> bool = default;
};

struct GameManifest {
    std::string name;
    std::string title;
    std::string game;
    std::string version;
    std::string entry;
    // A cooked scene (.lbf) relative to the data directory, opened before play starts; empty to use the game's own.
    std::string scene;
    // Extra content directories (`mount = "..."`, repeatable), relative to the project root, searched after the project's
    // own content and before the engine's. Development only: a packaged game carries its content cooked.
    std::vector<std::string> mounts;
    GiSettings gi;
    DiagnosticsSettings diagnostics;
    RenderSettings render;

    [[nodiscard]] static auto load(std::filesystem::path const &path, bool require_game = true) -> std::expected<GameManifest, std::string>;

    [[nodiscard]] auto to_text() const -> std::string;
};
