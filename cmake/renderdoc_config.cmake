function(configure_renderdoc target)
    set(options REQUIRED)
    set(one_value_args
        REAL_SOURCE
        STUB_SOURCE
        INCLUDE_CACHE_VARIABLE
    )
    set(multi_value_args)

    cmake_parse_arguments(
        DY_RENDERDOC
        "${options}"
        "${one_value_args}"
        "${multi_value_args}"
        ${ARGN}
    )

    if(NOT DY_RENDERDOC_REAL_SOURCE)
        message(FATAL_ERROR "configure_renderdoc: REAL_SOURCE is required")
    endif()

    if(NOT DY_RENDERDOC_STUB_SOURCE)
        message(FATAL_ERROR "configure_renderdoc: STUB_SOURCE is required")
    endif()

    if(NOT RENDERDOC_INCLUDE_CACHE_VARIABLE)
        set(RENDERDOC_INCLUDE_CACHE_VARIABLE RENDERDOC_INCLUDE_PATH)
    endif()

    option(ENABLE_RENDERDOC "Enable RenderDoc integration" ON)

    set(${RENDERDOC_INCLUDE_CACHE_VARIABLE}
        "${${RENDERDOC_INCLUDE_CACHE_VARIABLE}}"
        CACHE PATH
        "Path to directory containing renderdoc_app.h"
    )

    set(renderdoc_include_path "${${RENDERDOC_INCLUDE_CACHE_VARIABLE}}")
    set(renderdoc_available OFF)

    if(ENABLE_RENDERDOC)
        if(renderdoc_include_path)
            if(EXISTS "${renderdoc_include_path}/renderdoc_app.h")
                set(renderdoc_available ON)
            else()
                if(DY_RENDERDOC_REQUIRED)
                    message(FATAL_ERROR
                        "RenderDoc was required, but renderdoc_app.h was not found in "
                        "'${renderdoc_include_path}'."
                    )
                else()
                    message(WARNING
                        "RenderDoc requested, but renderdoc_app.h was not found in "
                        "'${renderdoc_include_path}'. RenderDoc integration disabled."
                    )
                endif()
            endif()
        else()
            find_path(renderdoc_include_path renderdoc_app.h)

            if(renderdoc_include_path)
                set(${RENDERDOC_INCLUDE_CACHE_VARIABLE}
                    "${renderdoc_include_path}"
                    CACHE PATH
                    "Path to directory containing renderdoc_app.h"
                    FORCE
                )
                set(renderdoc_available ON)
            elseif(DY_RENDERDOC_REQUIRED)
                message(FATAL_ERROR
                    "RenderDoc was required, but renderdoc_app.h was not found. "
                    "Set ${RENDERDOC_INCLUDE_CACHE_VARIABLE} to the directory "
                    "containing renderdoc_app.h."
                )
            endif()
        endif()
    endif()

    if(renderdoc_available)
        target_sources(${target} PRIVATE
            "${DY_RENDERDOC_REAL_SOURCE}"
        )

        set(renderdoc_isolated_dir "${CMAKE_BINARY_DIR}/renderdoc_isolated")

        file(MAKE_DIRECTORY "${renderdoc_isolated_dir}")

        configure_file(
            "${renderdoc_include_path}/renderdoc_app.h"
            "${renderdoc_isolated_dir}/renderdoc_app.h"
            COPYONLY
        )

        target_include_directories(${target} SYSTEM PRIVATE
            "${renderdoc_isolated_dir}"
        )

        # PUBLIC, not PRIVATE: application.cxx (engine_app) and
        # vulkan_bootstrap.cxx (a direct executable source) both
        # #include "gpu/renderdoc.hxx". Before the module split this define
        # reached them transitively for free (everything was one archive);
        # now that engine_gpu is its own library, PRIVATE would leave those
        # other targets seeing it undefined -- a silent bug, since
        # `#if HAS_RENDERDOC` evaluates false-via-undefined instead of
        # erroring.
        target_compile_definitions(${target} PUBLIC
            HAS_RENDERDOC=1
        )

        message(STATUS "RenderDoc: enabled, using ${renderdoc_include_path}")
    else()
        target_sources(${target} PRIVATE
            "${DY_RENDERDOC_STUB_SOURCE}"
        )

        target_compile_definitions(${target} PUBLIC
            HAS_RENDERDOC=0
        )

        message(STATUS "RenderDoc: disabled")
    endif()
endfunction()
