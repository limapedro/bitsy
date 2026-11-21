cmake_minimum_required(VERSION 3.20)

project(hagane
  VERSION 0.1.0
  DESCRIPTION "Hagane (鋼): a C11 deep learning framework with a Vulkan backend"
  LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(HG_BUILD_TESTS "Build the C unit tests" OFF)
option(HG_ENABLE_VULKAN "Enable the Vulkan backend (arrives in M2)" OFF)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra)
endif()

set(HG_CORE_SOURCES
  src/core/error.c
  src/core/dtype.c
  src/core/device.c
  src/core/shape.c
  src/core/storage.c
  src/core/tensor.c
  src/core/dispatcher.c
  src/core/init.c)

set(HG_CPU_REF_SOURCES
  src/backends/cpu_ref/add.c
  src/backends/cpu_ref/register.c)

set(HG_OP_SOURCES
  src/ops/add.c)

add_library(hagane SHARED
  ${HG_CORE_SOURCES}
  ${HG_CPU_REF_SOURCES}
  ${HG_OP_SOURCES})

target_include_directories(hagane
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_definitions(hagane PRIVATE HG_BUILD_SHARED)

# Keep the shared library next to test executables so ctypes and CTest can
# find it without extra path manipulation.
set_target_properties(hagane PROPERTIES
  OUTPUT_NAME hagane
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(HG_ENABLE_VULKAN)
  message(STATUS "Vulkan backend requested; shaders are compiled when added in M2")
  find_program(HG_GLSLC glslc)
  if(NOT HG_GLSLC)
    message(WARNING "glslc not found: shader compilation will be unavailable")
  endif()
endif()

if(HG_BUILD_TESTS)
  enable_testing()
  add_executable(hg_unit_tests tests/unit/test_core.c)
  target_link_libraries(hg_unit_tests PRIVATE hagane)
  add_test(NAME unit_core COMMAND hg_unit_tests)
endif()
