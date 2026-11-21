cmake_minimum_required(VERSION 3.20)

project(bitsy VERSION 0.2.0 LANGUAGES C
        DESCRIPTION "Bitsy: a C11 deep learning framework with a Vulkan backend")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(BT_BUILD_TESTS "Build the C unit tests" OFF)
option(BT_ENABLE_VULKAN "Enable the Vulkan compute backend" OFF)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra)
endif()

set(BT_SOURCES
  src/core/error.c src/core/dtype.c src/core/device.c src/core/shape.c
  src/core/storage.c src/core/tensor.c src/core/dispatcher.c src/core/init.c
  src/backends/cpu_ref/elementwise.c src/backends/cpu_ref/matmul.c
  src/backends/cpu_ref/register.c
  src/ops/elementwise.c src/ops/matmul.c src/ops/conv.c
  src/core/conv_dispatch.c src/backends/cpu_ref/conv.c)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(BitsyShader)

if(BT_ENABLE_VULKAN)
  find_package(Vulkan REQUIRED)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  set(SHADER_HEADERS "")
  foreach(SHADER elementwise matmul conv gemm_tiled)
    set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/vulkan/shaders/${SHADER}.comp)
    set(SHADER_SPV ${CMAKE_BINARY_DIR}/generated/${SHADER}.spv)
    set(SHADER_HEADER ${CMAKE_BINARY_DIR}/generated/${SHADER}.spv.h)
    add_custom_command(OUTPUT ${SHADER_SPV} ${SHADER_HEADER}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/generated
      COMMAND ${BT_GLSLC} --target-env=vulkan1.2 -O -o ${SHADER_SPV} ${SHADER_SRC}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/spv_to_header.py ${SHADER_SPV} ${SHADER_HEADER}
      DEPENDS ${SHADER_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/tools/spv_to_header.py
      COMMENT "Compiling ${SHADER}.comp to SPIR-V and embedding it"
      VERBATIM)
    list(APPEND SHADER_HEADERS ${SHADER_HEADER})
  endforeach()

  list(APPEND BT_SOURCES
    src/backends/vulkan/vk_device.c
    src/backends/vulkan/vk_allocator.c
    src/backends/vulkan/vk_ops.c
    ${SHADER_HEADERS})
endif()

add_library(bitsy SHARED ${BT_SOURCES})
target_include_directories(bitsy PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_include_directories(bitsy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(bitsy PRIVATE BT_BUILD_SHARED)

# expf/tanhf in cpu_ref need libm on Linux (MSVC provides them in the CRT).
if(UNIX)
  target_link_libraries(bitsy PRIVATE m)
endif()

# Put the DLL/.so next to the test binaries so ctypes and CTest find it.
set_target_properties(bitsy PROPERTIES
  OUTPUT_NAME bitsy
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(BT_ENABLE_VULKAN)
  target_compile_definitions(bitsy PRIVATE BT_ENABLE_VULKAN)
  target_include_directories(bitsy PRIVATE ${Vulkan_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/generated)
  target_link_libraries(bitsy PRIVATE Vulkan::Vulkan)
  message(STATUS "Vulkan backend enabled: ${BT_GLSLC}")
endif()

if(BT_BUILD_TESTS)
  enable_testing()
  add_executable(bt_unit_tests tests/unit/test_core.c)
  target_link_libraries(bt_unit_tests PRIVATE bitsy)
  add_test(NAME unit_core COMMAND bt_unit_tests)
endif()
