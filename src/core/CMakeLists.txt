# ------------------------------------------------------------------------------
# engine_core -- bottom layer: allocators, handles, logging, error plumbing,
# thread pool. No dependency on any other engine_<module> library.
# ------------------------------------------------------------------------------

add_library(
    engine_core
    STATIC
        error_describe.cxx
        fly_string.cxx
        logger.cxx
        memory_tracker.cxx
        thread_pool.cxx
)

target_include_directories(
    engine_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_core
    PUBLIC
        engine_deps
    PRIVATE
        engine_options
)

target_precompile_headers(
    engine_core
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include/core/pch.hxx"
)
