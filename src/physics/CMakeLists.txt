# ------------------------------------------------------------------------------
# engine_physics: Bullet wrapper (PhysicsWorld) and physics components.
# ------------------------------------------------------------------------------

add_library(
    engine_physics
    STATIC
        character_body.cxx
        mesh_collider.cxx
        physics_world.cxx
)

target_include_directories(
    engine_physics
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_physics
    PUBLIC
        engine_deps
        engine_assets
        engine_gpu
        engine_core
    PRIVATE
        engine_options
)

target_precompile_headers(
    engine_physics
    REUSE_FROM
        engine_core
)
