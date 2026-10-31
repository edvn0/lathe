set(
    engine_assets_sources
    assets_error_describe.cxx
    assimp_import.cxx
    gpu_texture_encoder.cxx
    gpu_texture_service.cxx
    geometry_allocator.cxx
    geometry_arena.cxx
    hdr_image.cxx
    load_model.cxx
    meshlet.cxx
    material_storage.cxx
    mesh_storage.cxx
    model_load_profile.cxx
    model_pick_geometry.cxx
    model_skin.cxx
    model_storage.cxx
    model_streamer.cxx
    primitive_meshes.cxx
    procedural_textures.cxx
    file_change_watcher.cxx
    shader_bake.cxx
    shader_pack.cxx
    shader_pack_cache.cxx
    slang_compiler.cxx
    stb.cxx
    texture_cache.cxx
    texture_pipeline.cxx
    texture_streamer.cxx
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
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
        engine_animation
        engine_gpu
        engine_storage
        engine_core
    PRIVATE
        engine_options
        spirv-tools::opt
)

# libktx ships its own zstd (the symbols are in the static library); the texture cache reads zstd-supercompressed
# levels with it directly instead of going through ktxTexture2.
target_include_directories(
    engine_assets
    PRIVATE
        "${ktx_SOURCE_DIR}/external/basisu/zstd"
)

# libktx's basisu headers are third-party code that does not build clean under -Werror -Wpedantic -Wconversion; as a system
# include their warnings are not ours.
target_include_directories(
    engine_assets
    SYSTEM
    PRIVATE
        "${ktx_SOURCE_DIR}/external/basisu"
)

# The BC7 benchmark decodes with the encoder library libktx carries, which is built with these.
target_compile_definitions(
    engine_assets
    PRIVATE
        BASISU_SUPPORT_SSE=1
        BASISD_SUPPORT_KTX2=1
)

target_precompile_headers(
    engine_assets
    REUSE_FROM
        engine_core
)
