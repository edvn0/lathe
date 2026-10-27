#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <expected>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <entt/entt.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "assets/material.hxx"
#include "assets/material_storage.hxx"
#include "assets/model.hxx"
#include "core/renderer_error.hxx"
#include "gpu/image_storage.hxx"
#include "rendering/game_graph.hxx"
#include "scene/components.hxx"
#include "terrain/planet.hxx"

struct Renderer;
class GameGpu;

// Where the camera is this frame, for content placed relative to it.
struct PlanetView {
    // The floating origin (CameraParams::origin): the world point render space is centred on.
    glm::dvec3 origin{0.0};
    // The camera in render space (normally near zero).
    glm::dvec3 camera{0.0};
    float time = 0.0F;
};

// Draws every entity's Components::Planet (docs/globe.md). Per planet it loads the elevation map and the colour on
// worker threads, keeps a cache of terrain and ocean chunks built on the thread pool from a cube-sphere quadtree
// chosen around the camera, and submits the chunks with camera-relative transforms worked out in double precision
// (render space is centred on CameraParams::origin), so nothing planet-sized reaches the GPU. The ocean is a second,
// smooth quadtree at sea level drawn with a water material in its spherical mode; the atmosphere is a compute pass
// after lighting (assets/shaders/atmosphere.slang).
class PlanetSystem {
public:
    // Chunks started on the thread pool per frame, chunks turned into models per frame, and chunks kept in all.
    static constexpr std::uint32_t builds_per_frame = 24;
    static constexpr std::size_t max_in_flight = 64;
    static constexpr std::uint32_t uploads_per_frame = 24;
    static constexpr std::chrono::microseconds upload_budget{4000};
    static constexpr std::size_t chunk_budget = 2400;
    // Frames a chunk is kept for after it was last drawn (it may come back as the camera turns).
    static constexpr std::uint64_t keep_frames = 120;
    static constexpr std::uint32_t terrain_cells = 32;
    static constexpr std::uint32_t ocean_cells = 16;
    static constexpr std::uint8_t ocean_max_level = 12;
    static constexpr std::uint32_t retire_frames = 4;

    PlanetSystem() = default;
    ~PlanetSystem();
    PlanetSystem(PlanetSystem const &) = delete;
    auto operator=(PlanetSystem const &) -> PlanetSystem & = delete;

    // Registers the atmosphere shader. Safe to call again.
    auto create(GameGpu &gpu) -> std::expected<void, RendererError>;

    // Brings the planets in `registry` up to date (loads, chunk builds) and submits them. Call once per frame with the
    // other scene submissions, on the render thread.
    auto submit(Renderer &renderer, entt::registry const &registry, PlanetView const &view) -> void;

    // Adds the atmosphere pass in GameSlot::after_lighting. `view` and `projection` are this frame's camera (render
    // space).
    auto declare(GameGraph &graph, entt::registry const &registry, glm::mat4 const &view, glm::mat4 const &projection,
                 glm::dvec3 const &origin, glm::vec3 const &sun_direction, float sun_intensity) -> void;

    // Releases every chunk, material and image now (the renderer must be idle).
    auto clear(Renderer &renderer) -> void;

    // How many terrain and ocean chunks were drawn last frame, and how many are cached or being built.
    struct Stats {
        std::size_t terrain_drawn = 0;
        std::size_t ocean_drawn = 0;
        std::size_t cached = 0;
        std::size_t building = 0;
        bool heightmap_ready = false;
    };
    [[nodiscard]] auto stats() const noexcept -> Stats { return stats_; }

    // The terrain's elevation (m above the radius, exaggeration applied) under a world point, or nothing when no
    // planet with a loaded heightmap is there. The nearest planet is used.
    [[nodiscard]] auto terrain_elevation(entt::registry const &registry, glm::dvec3 const &world) const
            -> std::optional<double>;
    // Whether a world point is inside a planet's ocean: below sea level and above the sea floor.
    [[nodiscard]] auto is_underwater(entt::registry const &registry, glm::dvec3 const &world) const -> bool;

    // The camera-relative model matrix of a chunk: planet frame -> world (`planet`) -> render space (minus `origin`).
    [[nodiscard]] static auto chunk_matrix(glm::dmat4 const &planet, planet::ChunkMesh const &chunk,
                                           glm::dvec3 const &origin) -> glm::mat4;

private:
    // A chunk built on the thread pool: its frame and its model data, ready to upload.
    struct Built;

    struct Chunk {
        std::future<std::shared_ptr<Built>> building;
        // The chunk's frame (origin, rotation, scale); its vertex lists are dropped once uploaded.
        std::shared_ptr<planet::ChunkMesh> mesh;
        ModelHandle model{};
        std::uint64_t last_used = 0;
    };

    using ChunkMap = std::unordered_map<planet::ChunkKey, Chunk, planet::ChunkKeyHash>;

    struct Body {
        Components::Planet planet;

        std::future<std::expected<planet::Heightmap, std::string>> heightmap_loading;
        std::shared_ptr<planet::Heightmap const> heightmap;
        std::future<std::expected<planet::ColourImage, std::string>> colour_loading;
        ImageHandle colour_image{};
        bool load_failed = false;

        MaterialHandle terrain_material{};
        MaterialCreateInfo terrain_info{};
        std::array<MaterialHandle, 16> debug_materials{};
        MaterialHandle ocean_material{};

        ChunkMap terrain;
        ChunkMap ocean;
        bool seen = false;
    };

    struct Retired {
        ModelHandle model{};
        MaterialHandle material{};
        ImageHandle image{};
        std::uint32_t frames_left = retire_frames;
    };

    auto start_loading(Body &body) -> void;
    auto poll_loading(Renderer &renderer, Body &body) -> void;
    auto ensure_materials(Renderer &renderer, Body &body) -> void;
    auto update_ocean_material(Renderer &renderer, Body &body, glm::dmat4 const &planet, glm::dvec3 const &camera,
                               PlanetView const &view) -> void;
    auto draw_tree(Renderer &renderer, Body &body, ChunkMap &chunks, std::vector<planet::ChunkKey> const &wanted,
                   bool ocean, glm::dmat4 const &planet, PlanetView const &view, std::uint32_t &builds,
                   std::uint32_t &uploads) -> std::size_t;
    auto drop_body(Body &body) -> void;
    auto age_retired(Renderer &renderer, bool everything) -> void;

    std::unordered_map<entt::entity, Body> bodies_;
    std::vector<Retired> retired_;
    std::uint64_t frame_ = 0;
    Stats stats_{};

    GameGpu *gpu_ = nullptr;
    std::optional<GameComputeShader> atmosphere_shader_;
};
