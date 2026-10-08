# ------------------------------------------------------------------------------
# engine_animation: skeletons, clips, blending, state machines and batched
# skinning-palette evaluation. Pure CPU; no Vulkan.
# ------------------------------------------------------------------------------

add_library(
    engine_animation
    STATIC
        batch.cxx
        clip.cxx
        clip_state_table.cxx
        humanoid.cxx
        joint_remap.cxx
        skeleton.cxx
        state_machine.cxx
)

target_include_directories(
    engine_animation
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_animation
    PUBLIC
        engine_deps
        engine_core
    PRIVATE
        engine_options
)

target_precompile_headers(
    engine_animation
    REUSE_FROM
        engine_core
)
