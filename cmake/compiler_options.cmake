# ------------------------------------------------------------------------------
# engine_options: shared compiler configuration (warnings, exceptions, RTTI,
# sanitizers) for every project-owned target -- module libraries, the
# executable, and tests.
#
# An INTERFACE library instead of a per-target function call, so the options
# are defined exactly once. That also matters for target_precompile_headers'
# REUSE_FROM (see the module library CMakeLists.txt files): reusing a PCH
# requires the reusing target to have identical compile flags to the target
# that compiled it, so every module needs the same engine_options.
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

          # The error-context types (ErrorContext and the *Error aggregates'
          # `.cause`/`.context` fields) are deliberately optional trailing
          # members with defaults -- callers only ever fill in what's
          # relevant to that failure. GCC's -Wmissing-field-initializers
          # fires on every designated-init that omits them even though a
          # default member initializer exists, which isn't a real bug.
          #
          # Formerly applied PRIVATE to mingw-vulkan-core only; folded in
          # here so it still applies now that project code is split across
          # multiple libraries.
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

  # ASan/UBSan need MinGW's runtime support, which is inconsistent across
  # cross-compilers, so sanitizers are only offered for native builds.
  #
  # -fno-sanitize=alignment: vendored stb_image_resize2 (pulled in by
  # texture_pipeline.cxx) does intentional misaligned uint64 accesses in
  # its SIMD-ish coefficient packing; harmless on x86/x86-64 and not
  # something this project can patch upstream, so alignment UB is excluded
  # while every other UBSan check stays active.
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
