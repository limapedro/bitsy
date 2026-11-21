cmake_minimum_required(VERSION 3.20)

project(bitsy VERSION 0.2.0 LANGUAGES C
        DESCRIPTION "Bitsy: a C11 deep learning framework with a Vulkan backend")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(BT_BUILD_TESTS "Build the C unit tests" OFF)
option(BT_ENABLE_VULKAN "Enable the Vulkan backend (arrives in M2)" OFF)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra)
endif()

# One library, one source list: no extra CMake indirection to audit.
add_library(bitsy SHARED
  src/core/error.c src/core/dtype.c src/core/device.c src/core/shape.c
  src/core/storage.c src/core/tensor.c src/core/dispatcher.c src/core/init.c
  src/backends/cpu_ref/add.c src/backends/cpu_ref/register.c
  src/ops/add.c)

target_include_directories(bitsy PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_include_directories(bitsy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(bitsy PRIVATE BT_BUILD_SHARED)

# Put the DLL/.so next to the test binaries so ctypes and CTest find it.
set_target_properties(bitsy PROPERTIES
  OUTPUT_NAME bitsy
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(BitsyShader)
if(BT_ENABLE_VULKAN)
  message(STATUS "Vulkan backend requested; shaders are compiled when added in M2")
endif()

if(BT_BUILD_TESTS)
  enable_testing()
  add_executable(bt_unit_tests tests/unit/test_core.c)
  target_link_libraries(bt_unit_tests PRIVATE bitsy)
  add_test(NAME unit_core COMMAND bt_unit_tests)
endif()
