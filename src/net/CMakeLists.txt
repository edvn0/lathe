# ------------------------------------------------------------------------------
# engine_net: HttpClient and its libcurl handler.
# ------------------------------------------------------------------------------

add_library(
    engine_net
    STATIC
        curl_http_handler.cxx
        http_client.cxx
        net_error_describe.cxx
)

target_include_directories(
    engine_net
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_net
    PUBLIC
        engine_deps
        engine_core
    PRIVATE
        engine_options
        CURL::libcurl
)

target_precompile_headers(
    engine_net
    REUSE_FROM
        engine_core
)
