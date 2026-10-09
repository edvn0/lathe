# ------------------------------------------------------------------------------
# engine_rendering: Renderer, render passes, Scene/Entity, debug/ImGui/terminal
# overlays.
# ------------------------------------------------------------------------------

add_library(
    engine_rendering
    STATIC
        cluster_grid.cxx
        debug_renderer.cxx
        editor_icons.cxx
        engine_models.cxx
        environment_panel.cxx
        environment.cxx
        file_browser.cxx
        frame_graph/aliasing.cxx
        frame_graph/compiler.cxx
        frame_graph/describe.cxx
        frame_graph/executor.cxx
        frame_graph/frame_graph.cxx
        frame_graph/layout.cxx
        frame_graph/resource_info.cxx
        frame_graph/pass_profiler.cxx
        frame_graph/scheduler.cxx
        frame_graph/transient_allocator.cxx
        frame_graph/vk_translate.cxx
        forward_target.cxx
        imgui_renderer.cxx
        overlay.cxx
        pipeline_graph_repository.cxx
        renderer.cxx
        renderer_frame_graph.cxx
        rendering_error_describe.cxx
        render_passes.cxx
        scene.cxx
        sky_model.cxx
        screenshot.cxx
        script_storage.cxx
        shadow_cascades.cxx
        terminal_widget.cxx
        toast.cxx
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
        # Kept private so the scripting and app modules can only reach toasts through gui::toast_*.
        imgui_notify
)

target_precompile_headers(
    engine_rendering
    REUSE_FROM
        engine_core
)

# renderer.cxx and render_passes.cxx include the generated
# shader_push_constants.hxx.
add_dependencies(
    engine_rendering
    generate_shader_push_constants
)
