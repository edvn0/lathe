# ------------------------------------------------------------------------------
# engine_serialisation: the Lathe Binary Format (.lbf) -- the chunked container,
# cooked model/texture codecs, asset packs, the scene codec, and saving/loading a
# Scene through the Renderer. Sits above engine_rendering because Scene lives
# there and loading a scene creates GPU resources.
# ------------------------------------------------------------------------------

add_library(
    engine_serialisation
    STATIC
        asset_id.cxx
        asset_pack.cxx
        checksum.cxx
        cooked_model.cxx
        cooked_environment.cxx
        cooked_texture.cxx
        lbf_container.cxx
        scene_codec.cxx
        scene_serialisation.cxx
        serialisation_error_describe.cxx
)

target_include_directories(
    engine_serialisation
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

# zstd comes from the copy libktx already compiles in (external/basisu/zstd).
# Linking a second zstd would define every ZSTD_* symbol twice.
target_include_directories(
    engine_serialisation
    SYSTEM
    PRIVATE
        "${ktx_SOURCE_DIR}/external/basisu/zstd"
)

target_link_libraries(
    engine_serialisation
    PUBLIC
        engine_deps
        engine_rendering
        engine_scene
        engine_assets
        engine_gpu
        engine_core
    PRIVATE
        engine_options
)

target_precompile_headers(
    engine_serialisation
    REUSE_FROM
        engine_core
)
