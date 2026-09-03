set(
    engine_assets_sources
    assets_error_describe.cxx
    geometry_allocator.cxx
    geometry_arena.cxx
    load_model.cxx
    material_storage.cxx
    mesh_storage.cxx
    model_storage.cxx
    model_streamer.cxx
    primitive_meshes.cxx
    shader_hot_reload_watcher.cxx
    slang_compiler.cxx
    stb.cxx
    texture_pipeline.cxx
    texture_streamer.cxx
)

# Exactly one platform's slang_library.cxx is compiled in, matching the
# platform this project is being built for -- not the host running CMake
# (relevant when cross-compiling windows-mingw from a Linux host).
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  list(APPEND engine_assets_sources platform/windows/slang_library.cxx)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND engine_assets_sources platform/linux/slang_library.cxx)
endif()

add_library(
    engine_assets
    STATIC
        ${engine_assets_sources}
)

target_include_directories(
    engine_assets
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_assets
    PUBLIC
        engine_deps
        engine_gpu
        engine_core
    PRIVATE
        engine_options
        spirv-tools::opt
)

target_precompile_headers(
    engine_assets
    REUSE_FROM
        engine_core
)
