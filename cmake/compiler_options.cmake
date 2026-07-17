# ------------------------------------------------------------------------------
# Compiler-specific warnings and exception/RTTI configuration
#
# MSVC only appears on Windows. GNU/Clang appears on both Windows (MinGW) and
# Linux, so this stays compiler-gated rather than platform-gated.
# ------------------------------------------------------------------------------

function(configure_compiler_options target_name)
  if(MSVC)
    target_compile_options(
            ${target_name}
            PRIVATE
                /W4
                /permissive-
                /Zc:preprocessor
                /Zc:__cplusplus
        )

    if(MINGW_VULKAN_ENABLE_EXCEPTIONS)
      target_compile_options(
                ${target_name}
                PRIVATE
                    /EHsc
                    /GR-
            )
    else()
      target_compile_options(
                ${target_name}
                PRIVATE
                    /EHs-c-
                    /GR-
            )

      target_compile_definitions(
                ${target_name}
                PRIVATE
                    _HAS_EXCEPTIONS=0
            )
    endif()

  elseif(
        CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
        OR CMAKE_CXX_COMPILER_ID MATCHES "Clang"
    )
    target_compile_options(
            ${target_name}
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
                -Wconversion
                -Wsign-conversion
                -Wshadow
                -Wnon-virtual-dtor
                -Wold-style-cast
        )

    if(MINGW_VULKAN_ENABLE_EXCEPTIONS)
      target_compile_options(
                ${target_name}
                PRIVATE
                    -fexceptions
                    -fno-rtti
            )
    else()
      target_compile_options(
                ${target_name}
                PRIVATE
                    -fno-exceptions
                    -fno-rtti
            )
    endif()
  endif()

  if(NOT MINGW_VULKAN_ENABLE_EXCEPTIONS)
    target_compile_definitions(
            ${target_name}
            PRIVATE
                SPDLOG_NO_EXCEPTIONS
        )
  endif()
endfunction()
