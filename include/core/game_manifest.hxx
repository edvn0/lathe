#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>

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
    // The point lights a model gets from its emissive meshes when it loads: "auto" (none while the probes light them
    // themselves), "on" or "off".
    std::string synthesize_lights = "auto";

    auto operator==(GiSettings const &) const -> bool = default;
};

struct GameManifest {
    std::string name;
    std::string title;
    std::string game;
    std::string version;
    std::string entry;
    // A cooked scene (.lbf) relative to the data directory, opened before play starts; empty to use the game's own.
    std::string scene;
    GiSettings gi;

    [[nodiscard]] static auto load(std::filesystem::path const &path, bool require_game = true) -> std::expected<GameManifest, std::string>;

    [[nodiscard]] auto to_text() const -> std::string;
};
