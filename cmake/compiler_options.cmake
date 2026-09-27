# ------------------------------------------------------------------------------
# engine_options: shared compiler configuration (warnings, exceptions, RTTI,
# sanitizers) for every project target. Identical flags everywhere are also
# what lets targets share a PCH through REUSE_FROM.
# ------------------------------------------------------------------------------

add_library(engine_options INTERFACE)

if(MSVC)
  target_compile_options(
      engine_options
      INTERFACE
          /W4
          /permissive-
          /Zc:preprocessor
          /Zc:__cplusplus
  )

  if(MINGW_VULKAN_WERROR)
    target_compile_options(engine_options INTERFACE /WX)
  endif()

  if(MINGW_VULKAN_ENABLE_EXCEPTIONS)
    target_compile_options(
        engine_options
        INTERFACE
            /EHsc
            /GR-
    )
  else()
    target_compile_options(
        engine_options
        INTERFACE
            /EHs-c-
            /GR-
    )

    target_compile_definitions(
        engine_options
        INTERFACE
            _HAS_EXCEPTIONS=0
    )
  endif()

elseif(
      CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
      OR CMAKE_CXX_COMPILER_ID MATCHES "Clang"
  )
  target_compile_options(
      engine_options
      INTERFACE
          -Wall
          -Wextra
          -Wpedantic
          -Wconversion
          -Wshadow
          -Wnon-virtual-dtor

          # Error types have optional trailing members with defaults, which
          # -Wmissing-field-initializers flags on every designated initializer.
          -Wno-missing-field-initializers
          -Wno-old-style-cast
  )

  if(MINGW_VULKAN_WERROR)
    target_compile_options(engine_options INTERFACE -Werror)
  endif()

  if(MINGW_VULKAN_ENABLE_EXCEPTIONS)
    target_compile_options(
        engine_options
        INTERFACE
            -fexceptions
            -fno-rtti
    )
  else()
    target_compile_options(
        engine_options
        INTERFACE
            -fno-exceptions
            -fno-rtti
    )
  endif()

  # Sanitizers are native-only; MinGW's runtime support is inconsistent.
  #
  # -fno-sanitize=alignment: stb_image_resize2 does deliberate misaligned
  # accesses.
  if(MINGW_VULKAN_SANITIZE AND NOT MINGW)
    target_compile_options(
        engine_options
        INTERFACE
            -fsanitize=address,undefined
            -fno-sanitize=alignment
            -fno-sanitize-recover=all
            -fno-omit-frame-pointer
    )
    target_link_options(
        engine_options
        INTERFACE
            -fsanitize=address,undefined
    )
  endif()
endif()

if(NOT MINGW_VULKAN_ENABLE_EXCEPTIONS)
  target_compile_definitions(
      engine_options
      INTERFACE
          SPDLOG_NO_EXCEPTIONS
  )
endif()
