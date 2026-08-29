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

    if(MINGW_VULKAN_WERROR)
      target_compile_options(${target_name} PRIVATE /WX)
    endif()

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
                -Wshadow
                -Wnon-virtual-dtor
        )

    if(MINGW_VULKAN_WERROR)
      target_compile_options(${target_name} PRIVATE -Werror)
    endif()

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
              ${target_name}
              PRIVATE
                  -fsanitize=address,undefined
                  -fno-sanitize=alignment
                  -fno-sanitize-recover=all
                  -fno-omit-frame-pointer
          )
      target_link_options(
              ${target_name}
              PRIVATE
                  -fsanitize=address,undefined
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
