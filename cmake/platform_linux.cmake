# ------------------------------------------------------------------------------
# Linux target configuration
# ------------------------------------------------------------------------------

function(configure_linux_target target_name slang_lib_dir slang_bin_dir)
  file(
        GLOB slang_runtime_files
        CONFIGURE_DEPENDS
        "${slang_lib_dir}/*.so"
        "${slang_lib_dir}/*.so.*"
        "${slang_bin_dir}/*.so"
        "${slang_bin_dir}/*.so.*"
    )

  if(NOT slang_runtime_files)
    message(
            FATAL_ERROR
            "No Slang shared object files found in "
            "${slang_lib_dir} or ${slang_bin_dir}"
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
endfunction()
