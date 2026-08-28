file(
    GLOB test_sources
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*_test.cxx"
)

add_executable(
    mingw-vulkan-tests
    ${test_sources}
)

target_include_directories(
    mingw-vulkan-tests
    PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
)

target_compile_definitions(
    mingw-vulkan-tests
    PRIVATE
        DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
        TEST_ASSETS_DIR="${PROJECT_SOURCE_DIR}"
)

target_link_libraries(
    mingw-vulkan-tests
    PRIVATE
        mingw-vulkan-core
        doctest::doctest_with_main
)

configure_compiler_options(mingw-vulkan-tests)

doctest_discover_tests(
    mingw-vulkan-tests
    ADD_LABELS 1
)
