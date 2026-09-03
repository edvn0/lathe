# ------------------------------------------------------------------------------
# engine_rendering -- Renderer, render passes, Scene/Entity (see
# MODULARIZATION_HANDOFF.md's "Scene/Entity reclassification" note for why
# those two live here and not in engine_scene), debug/imgui/terminal overlays.
# ------------------------------------------------------------------------------

add_library(
    engine_rendering
    STATIC
        debug_renderer.cxx
        engine_models.cxx
        forward_target.cxx
        imgui_renderer.cxx
        pipeline_graph_repository.cxx
        renderer.cxx
        rendering_error_describe.cxx
        render_passes.cxx
        scene.cxx
        screenshot.cxx
        script_storage.cxx
        shadow_cascades.cxx
        terminal_widget.cxx
)

target_include_directories(
    engine_rendering
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_rendering
    PUBLIC
        engine_deps
        engine_terrain
        engine_scene
        engine_physics
        engine_assets
        engine_gpu
        engine_core
    PRIVATE
        engine_options
)

target_precompile_headers(
    engine_rendering
    REUSE_FROM
        engine_core
)

# renderer.cxx and render_passes.cxx are the only two includers of the
# generated shader_push_constants.hxx; the include dir itself comes for free
# via engine_deps, but the codegen must still be ordered before this library
# builds.
add_dependencies(
    engine_rendering
    generate_shader_push_constants
)
