# ------------------------------------------------------------------------------
# engine_storage: CacheStore, the SQLite-indexed on-disk cache shared by every
# process. Depends on engine_core only so any module above it can cache.
# ------------------------------------------------------------------------------

add_library(
    engine_storage
    STATIC
        cache_store.cxx
        caches.cxx
        storage_error_describe.cxx
)

target_include_directories(
    engine_storage
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_storage
    PUBLIC
        engine_deps
        engine_core
    PRIVATE
        engine_options
        sqlite3
)

target_precompile_headers(
    engine_storage
    REUSE_FROM
        engine_core
)
