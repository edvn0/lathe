# ------------------------------------------------------------------------------
# engine_core: allocators, handles, logging, errors, thread pool.
# ------------------------------------------------------------------------------

add_library(
    engine_core
    STATIC
        command_line.cxx
        error_describe.cxx
        fly_string.cxx
        human_readable_bytes.cxx
        json.cxx
        logger.cxx
        memory_tracker.cxx
        random.cxx
        sha256.cxx
        thread_pool.cxx
)

target_include_directories(
    engine_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

# Allows SCHED_IDLE worker threads. PUBLIC because BS_thread_pool.hpp must see
# the same definition in every translation unit.
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
        OpenSSL::Crypto
)

target_precompile_headers(
    engine_core
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include/core/pch.hxx"
)
