# GPU crash dumps through the NVIDIA Nsight Aftermath SDK (docs/gpu-crash-dumps.md). The SDK is not part of the repository
# (its licence does not let it be vendored): point AFTERMATH_SDK_DIR at an extracted copy (the directory with include/ and
# lib/x64/). Without it the build uses stub.cxx and collects nothing.
function(configure_aftermath target)
    set(options)
    set(one_value_args REAL_SOURCE STUB_SOURCE)
    set(multi_value_args)

    cmake_parse_arguments(DY_AFTERMATH "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT DY_AFTERMATH_REAL_SOURCE OR NOT DY_AFTERMATH_STUB_SOURCE)
        message(FATAL_ERROR "configure_aftermath: REAL_SOURCE and STUB_SOURCE are required")
    endif()

    set(AFTERMATH_SDK_DIR "${AFTERMATH_SDK_DIR}" CACHE PATH
        "Nsight Aftermath SDK (the directory holding include/ and lib/x64/)")
    option(ENABLE_AFTERMATH "GPU crash dumps through the Nsight Aftermath SDK, when it is found" ON)

    set(aftermath_header "${AFTERMATH_SDK_DIR}/include/GFSDK_Aftermath_GpuCrashDump.h")
    set(aftermath_library "${AFTERMATH_SDK_DIR}/lib/x64/libGFSDK_Aftermath_Lib.x64.so")

    if(ENABLE_AFTERMATH AND AFTERMATH_SDK_DIR AND EXISTS "${aftermath_header}" AND EXISTS "${aftermath_library}")
        target_sources(${target} PRIVATE "${DY_AFTERMATH_REAL_SOURCE}")
        target_include_directories(${target} SYSTEM PRIVATE "${AFTERMATH_SDK_DIR}/include")
        # The library is opened at run time (dlopen), from this path first.
        target_compile_definitions(${target} PRIVATE "LATHE_AFTERMATH_LIB=\"${aftermath_library}\"")
        target_compile_definitions(${target} PUBLIC HAS_AFTERMATH=1)
        message(STATUS "Nsight Aftermath: enabled, using ${AFTERMATH_SDK_DIR}")
    else()
        target_sources(${target} PRIVATE "${DY_AFTERMATH_STUB_SOURCE}")
        target_compile_definitions(${target} PUBLIC HAS_AFTERMATH=0)
        if(AFTERMATH_SDK_DIR AND ENABLE_AFTERMATH)
            message(WARNING "AFTERMATH_SDK_DIR '${AFTERMATH_SDK_DIR}' does not hold the Nsight Aftermath SDK; GPU crash dumps disabled")
        else()
            message(STATUS "Nsight Aftermath: disabled (set AFTERMATH_SDK_DIR to enable GPU crash dumps)")
        endif()
    endif()
endfunction()
