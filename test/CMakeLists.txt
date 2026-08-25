add_executable(
    fly_string_tests
    fly_string_test.cxx
)

target_include_directories(
    fly_string_tests
    PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
)

target_compile_definitions(fly_string_tests PRIVATE DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS)

target_link_libraries(
    fly_string_tests
    PRIVATE
        mingw-vulkan-core
        doctest::doctest_with_main
        spdlog::spdlog
        BS_thread_pool
)

configure_compiler_options(fly_string_tests)

doctest_discover_tests(
    fly_string_tests
    ADD_LABELS 1
)