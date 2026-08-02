function(configure_windows_target target_name slang_bin_dir)
  target_compile_definitions(
        ${target_name}
        PRIVATE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
            UNICODE
            _UNICODE
    )

  if(MINGW)
    target_link_options(
            ${target_name}
            PRIVATE
                -static-libgcc
                -static-libstdc++
        )
  endif()

  # -- Slang runtime deployment (DLLs) --

  file(
        GLOB slang_runtime_files
        CONFIGURE_DEPENDS
        "${slang_bin_dir}/*.dll"
    )

  if(NOT slang_runtime_files)
    message(
            FATAL_ERROR
            "No Slang DLL files found in ${slang_bin_dir}"
        )
  endif()

  foreach(slang_runtime_file IN LISTS slang_runtime_files)
    add_custom_command(
            TARGET ${target_name}
            POST_BUILD
            COMMAND
                "${CMAKE_COMMAND}" -E copy_if_different
                "${slang_runtime_file}"
                "$<TARGET_FILE_DIR:${target_name}>"
            VERBATIM
        )
  endforeach()

  set_target_properties(
        ${target_name}
        PROPERTIES
            VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${target_name}>"
    )
endfunction()
