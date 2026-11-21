# GLSL compute shaders are compiled to SPIR-V at build time.
#
# Usage:
#   include(cmake/HaganeShader.cmake)
#   hg_add_shader(hagane shaders/add.comp)
#
# The compiled .spv is produced next to the build tree and the target depends
# on it. No shader is built until the Vulkan backend adds one in M2.

find_program(HG_GLSLC NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin")

function(hg_add_shader TARGET SHADER_PATH)
  if(NOT HG_GLSLC)
    message(FATAL_ERROR
      "glslc was not found; install the Vulkan SDK to compile ${SHADER_PATH}")
  endif()

  get_filename_component(_stem "${SHADER_PATH}" NAME_WE)
  set(_source "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_PATH}")
  set(_output "${CMAKE_CURRENT_BINARY_DIR}/${_stem}.spv")

  add_custom_command(
    OUTPUT "${_output}"
    COMMAND "${HG_GLSLC}" --target-env=vulkan1.2 -O -o "${_output}" "${_source}"
    DEPENDS "${_source}"
    COMMENT "Compiling ${SHADER_PATH} to ${_stem}.spv"
    VERBATIM)

  add_custom_target(hg_shader_${_stem} DEPENDS "${_output}")
  add_dependencies(${TARGET} hg_shader_${_stem})
  set(HG_LAST_SHADER_OUTPUT "${_output}" PARENT_SCOPE)
endfunction()
