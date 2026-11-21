cmake_minimum_required(VERSION 3.20)
project(togishi_calibrate LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_program(GLSLC NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin")
if(NOT GLSLC)
  message(FATAL_ERROR "glslc required for the calibration shaders")
endif()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/shaders")
foreach(SHADER fill copy copy4 fma fma8 fma8u fma8u64 fma16 f16fma shared readlat empty)
  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/shaders/${SHADER}.spv"
    COMMAND ${GLSLC} --target-env=vulkan1.2 -O
            -o "${CMAKE_BINARY_DIR}/shaders/${SHADER}.spv"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}.comp"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}.comp"
    VERBATIM)
endforeach()

add_executable(perf_calibrate perf_calibrate.c
  ${CMAKE_BINARY_DIR}/shaders/fill.spv
  ${CMAKE_BINARY_DIR}/shaders/copy.spv
  ${CMAKE_BINARY_DIR}/shaders/copy4.spv
  ${CMAKE_BINARY_DIR}/shaders/fma.spv
  ${CMAKE_BINARY_DIR}/shaders/fma8.spv
  ${CMAKE_BINARY_DIR}/shaders/fma16.spv
  ${CMAKE_BINARY_DIR}/shaders/fma8u.spv
  ${CMAKE_BINARY_DIR}/shaders/fma8u64.spv
  ${CMAKE_BINARY_DIR}/shaders/f16fma.spv
  ${CMAKE_BINARY_DIR}/shaders/shared.spv
  ${CMAKE_BINARY_DIR}/shaders/readlat.spv
  ${CMAKE_BINARY_DIR}/shaders/empty.spv)
target_include_directories(perf_calibrate PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(perf_calibrate PRIVATE Vulkan::Vulkan)
if(UNIX)
  target_link_libraries(perf_calibrate PRIVATE m)
endif()
