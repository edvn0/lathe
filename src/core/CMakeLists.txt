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

# Lets thread_pool.cxx set background worker threads to SCHED_IDLE (see
# BS::this_thread::set_os_thread_priority) so they never starve the
# render/main thread of CPU time regardless of core count or what else is
# running on the system -- PUBLIC so every consumer of engine_core's headers
# sees the same BS_thread_pool.hpp feature set (this must be defined
# identically in every translation unit that includes that header).
target_compile_definitions(
    engine_core
    PUBLIC
        BS_THREAD_POOL_NATIVE_EXTENSIONS
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
