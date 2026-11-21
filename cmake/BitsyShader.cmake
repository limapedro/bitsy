# GLSL compute shaders are compiled to SPIR-V at build time.
#
# Usage:
#   include(cmake/BitsyShader.cmake)
#   bt_add_shader(bitsy shaders/add.comp)
#
# The compiled .spv is produced next to the build tree and the target depends
# on it. No shader is built until the Vulkan backend adds one in M2.

find_program(BT_GLSLC NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin")

function(bt_add_shader TARGET SHADER_PATH)
  if(NOT BT_GLSLC)
    message(FATAL_ERROR
      "glslc was not found; install the Vulkan SDK to compile ${SHADER_PATH}")
  endif()

  get_filename_component(_stem "${SHADER_PATH}" NAME_WE)
  set(_source "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_PATH}")
  set(_output "${CMAKE_CURRENT_BINARY_DIR}/${_stem}.spv")

  add_custom_command(
    OUTPUT "${_output}"
    COMMAND "${BT_GLSLC}" --target-env=vulkan1.2 -O -o "${_output}" "${_source}"
    DEPENDS "${_source}"
    COMMENT "Compiling ${SHADER_PATH} to ${_stem}.spv"
    VERBATIM)

  add_custom_target(bt_shader_${_stem} DEPENDS "${_output}")
  add_dependencies(${TARGET} bt_shader_${_stem})
  set(BT_LAST_SHADER_OUTPUT "${_output}" PARENT_SCOPE)
endfunction()
