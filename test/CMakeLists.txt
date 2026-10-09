file(
    GLOB test_sources
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*_test.cxx"
)

add_executable(
    lathe-tests
    EXCLUDE_FROM_ALL
    ${test_sources}
    "${PROJECT_SOURCE_DIR}/game/src/chess_engine.cxx"
    "${PROJECT_SOURCE_DIR}/game/src/chess_server_game.cxx"
    "${PROJECT_SOURCE_DIR}/game/src/chess_lua.cxx"
    "${PROJECT_SOURCE_DIR}/game/src/net_lua.cxx"
)

target_include_directories(
    lathe-tests
    PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/game/include"
)

target_compile_definitions(
    lathe-tests
    PRIVATE
        DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
        TEST_ASSETS_DIR="${PROJECT_SOURCE_DIR}"
)

target_link_libraries(
    lathe-tests
    PRIVATE
        lathe::core
        engine_gameserver_ix
        engine_gameclient_ix
        doctest::doctest_with_main
        engine_options
)

# No ADD_LABELS: it relaunches the test binary once per test case to look up its
# suite, which added ~3.5 minutes to every link.
doctest_discover_tests(lathe-tests)
