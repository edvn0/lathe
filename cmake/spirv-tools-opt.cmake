function(find_spirv_tools_opt)
  if(TARGET spirv-tools::opt)
    return()
  endif()

  find_path(SpirvTools_INCLUDE_DIR NAMES spirv-tools/optimizer.hpp REQUIRED)
  find_library(SpirvTools_LIBRARY NAMES SPIRV-Tools REQUIRED)
  find_library(SpirvToolsOpt_LIBRARY NAMES SPIRV-Tools-opt REQUIRED)

  mark_as_advanced(SpirvTools_INCLUDE_DIR SpirvTools_LIBRARY SpirvToolsOpt_LIBRARY)

  add_library(spirv-tools::core STATIC IMPORTED)
  set_target_properties(
      spirv-tools::core
      PROPERTIES
          IMPORTED_LOCATION "${SpirvTools_LIBRARY}"
          INTERFACE_INCLUDE_DIRECTORIES "${SpirvTools_INCLUDE_DIR}"
  )

  add_library(spirv-tools::opt STATIC IMPORTED)
  set_target_properties(
      spirv-tools::opt
      PROPERTIES
          IMPORTED_LOCATION "${SpirvToolsOpt_LIBRARY}"
          INTERFACE_INCLUDE_DIRECTORIES "${SpirvTools_INCLUDE_DIR}"
  )

  target_link_libraries(spirv-tools::opt INTERFACE spirv-tools::core)
endfunction()