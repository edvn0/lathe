# ------------------------------------------------------------------------------
# MikkTSpace tangent-space generation library
#
# Public-domain single-file C library with no upstream CMakeLists, so we
# fetch the source via CPM and hand-roll a static target for it.
# ------------------------------------------------------------------------------

function(add_mikktspace_dependency)
    cpmaddpackage(
        NAME mikktspace
        GITHUB_REPOSITORY mmikk/MikkTSpace
        GIT_TAG 3e895b49d05ea07e4c2133156cfa94369e19e409
        DOWNLOAD_ONLY YES
    )

    if(NOT mikktspace_ADDED)
        message(
            FATAL_ERROR
            "Failed to fetch MikkTSpace via CPM"
        )
    endif()

    add_library(
        mikktspace_library
        STATIC
        "${mikktspace_SOURCE_DIR}/mikktspace.c"
        "${mikktspace_SOURCE_DIR}/mikktspace.h"
    )

    add_library(mikktspace::mikktspace ALIAS mikktspace_library)

    target_include_directories(
        mikktspace_library
        SYSTEM
        PUBLIC
        "${mikktspace_SOURCE_DIR}"
    )

    set_target_properties(
        mikktspace_library
        PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )

    # Upstream ships with -Wall-hostile code (implicit conversions,
    # comparison warnings); suppress rather than fight it, matching how
    # vma.cxx is already handled below.
    if(MSVC)
        target_compile_options(mikktspace_library PRIVATE /W0)
    elseif(
        CMAKE_C_COMPILER_ID STREQUAL "GNU"
        OR CMAKE_C_COMPILER_ID MATCHES "Clang"
    )
        target_compile_options(mikktspace_library PRIVATE -w)
    endif()
endfunction()
