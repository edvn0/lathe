# ------------------------------------------------------------------------------
# engine_terrain -- quadtree streaming, chunk meshing, noise field.
# ------------------------------------------------------------------------------

add_library(
    engine_terrain
    STATIC
        noise.cxx
        terrain_chunk.cxx
        terrain_mesh.cxx
        terrain_quadtree.cxx
        terrain_slot_pool.cxx
        terrain_world.cxx
)

target_include_directories(
    engine_terrain
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_terrain
    PUBLIC
        engine_deps
        engine_scene
        engine_physics
        engine_assets
        engine_gpu
        engine_core
    PRIVATE
        engine_options
)

target_precompile_headers(
    engine_terrain
    REUSE_FROM
        engine_core
)
