# ------------------------------------------------------------------------------
# engine_scripting: sandboxed Lua (ScriptEngine) and the "Script" panel. Runs against an entt::registry plus a
# hierarchy revision (ScriptWorld), not Scene, so tests need no Renderer.
# ------------------------------------------------------------------------------

add_library(
    engine_scripting
    STATIC
        entity_index.cxx
        lua_allocator.cxx
        lua_runtime.cxx
        script_bindings.cxx
        script_engine.cxx
        script_error.cxx
        script_widget.cxx
)

target_include_directories(engine_scripting PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../../include")

target_link_libraries(
    engine_scripting
    PUBLIC
        engine_deps
        engine_rendering
        engine_terrain
        engine_scene
        engine_physics
        engine_assets
        engine_gpu
        engine_core
    PRIVATE
        engine_options
)

target_precompile_headers(engine_scripting REUSE_FROM engine_core)
