# ------------------------------------------------------------------------------
# engine_physics -- Bullet3 wrapper (PhysicsWorld) and physics ECS components.
# Depends on engine_assets/gpu/core purely by the shared-layering convention;
# see the root CMakeLists.txt comment on engine_deps for why every module
# links every layer below it.
# ------------------------------------------------------------------------------

add_library(
    engine_physics
    STATIC
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
