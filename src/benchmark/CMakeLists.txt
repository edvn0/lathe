# ------------------------------------------------------------------------------
# engine_benchmark: the --benchmark runner, its scenarios (including the shadow city stress scene), analysis and
# comparison. Sits above engine_app because the driver runs the Application headless.
# ------------------------------------------------------------------------------

add_library(
    engine_benchmark
    STATIC
        benchmark.cxx
        benchmark_analysis.cxx
        benchmark_compare.cxx
        benchmark_driver.cxx
        benchmark_environment.cxx
        benchmark_scenarios.cxx
)

# Recorded in benchmark results (benchmark/benchmark_environment.hxx): the build type, and where to read the source
# revision from.
target_compile_definitions(
    engine_benchmark
    PRIVATE
        LATHE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        LATHE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        LATHE_TRACK_MEMORY=$<BOOL:${lathe_track_memory}>
)

target_include_directories(
    engine_benchmark
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_benchmark
    PUBLIC
        engine_app
        engine_game
        engine_deps
        engine_rendering
        engine_terrain
        engine_scene
        engine_assets
        engine_gpu
        engine_core
    PRIVATE
        engine_options
)

target_precompile_headers(
    engine_benchmark
    REUSE_FROM
        engine_core
)
