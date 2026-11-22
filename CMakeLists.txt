cmake_minimum_required(VERSION 3.20)

project(bitsy VERSION 0.2.0 LANGUAGES C
        DESCRIPTION "Bitsy: a C11 deep learning framework with a Vulkan backend")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(BT_BUILD_TESTS "Build the C unit tests" OFF)
option(BT_ENABLE_VULKAN "Enable the Vulkan compute backend" OFF)
option(BT_BUILD_COMPACT "Build the 0.2 compact lazy runtime (bitsy_rt)" ON)
option(BT_COMPACT_VULKAN "Enable Vulkan in the compact runtime" OFF)

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
  src/ops/elementwise.c src/ops/matmul.c src/ops/conv.c src/ops/sgd.c src/ops/reduce.c
  src/core/conv_dispatch.c src/core/profiler.c src/backends/cpu_ref/conv.c)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(BitsyShader)

if(BT_ENABLE_VULKAN)
  find_package(Vulkan REQUIRED)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  set(SHADER_HEADERS "")
  foreach(SHADER elementwise elementwise_vec4 matmul conv gemm_tiled gemm_rb gemm_rb_vec4 gemm_rb_vec4_k16 gemm_rb_vec4_swz gemm_rb_vec4_128x64 gemm_rb_vec4_64x128 gemm_bias gemm_tiled_bias log_softmax ce_loss splitk_reduce fused sgd_step reduce_sum tanh_backward broadcast_grad dual_mul sgd_fused gemm_tiled_bias_tanh gelu)
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

if(BT_BUILD_COMPACT)
  set(RT_SOURCES src/rt/rt.c src/rt/rt_vk.c)
  add_library(bitsy_rt SHARED ${RT_SOURCES})
  target_include_directories(bitsy_rt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include/bitsy_rt)
  target_compile_definitions(bitsy_rt PRIVATE BITSY_RT_BUILD_SHARED)
  set_target_properties(bitsy_rt PROPERTIES
    OUTPUT_NAME bitsy_rt
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  if(UNIX)
    target_link_libraries(bitsy_rt PRIVATE m)
  endif()
  if(BT_COMPACT_VULKAN)
    find_package(Vulkan REQUIRED)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(RT_SHADERS fused gemm reduce)
    foreach(S ${RT_SHADERS})
      set(SPV ${CMAKE_BINARY_DIR}/generated/rt_${S}.spv)
      set(HDR ${CMAKE_BINARY_DIR}/generated/rt_${S}.spv.h)
      add_custom_command(OUTPUT ${SPV} ${HDR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/generated
        COMMAND ${BT_GLSLC} --target-env=vulkan1.2 -O -o ${SPV} ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/shaders/${S}.comp
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/spv_to_header.py ${SPV} ${HDR}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/shaders/${S}.comp
        COMMENT "Compiling rt ${S}.comp to SPIR-V"
        VERBATIM)
      list(APPEND RT_SHADER_HEADERS ${HDR})
    endforeach()
    target_sources(bitsy_rt PRIVATE ${RT_SHADER_HEADERS})
    target_compile_definitions(bitsy_rt PRIVATE RT_WITH_VULKAN)
    target_include_directories(bitsy_rt PRIVATE ${Vulkan_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/generated)
    target_link_libraries(bitsy_rt PRIVATE Vulkan::Vulkan)
    message(STATUS "Compact runtime Vulkan enabled: ${BT_GLSLC}")
  endif()
endif()

if(BT_BUILD_TESTS)
  enable_testing()
  add_executable(bt_unit_tests tests/unit/test_core.c)
  target_link_libraries(bt_unit_tests PRIVATE bitsy)
  add_test(NAME unit_core COMMAND bt_unit_tests)
  if(BT_BUILD_COMPACT)
    add_executable(rt_unit_tests tests/unit/test_rt.c)
    target_link_libraries(rt_unit_tests PRIVATE bitsy_rt)
    add_executable(rt_unit_tests2 tests/unit/test_rt2.c)
    target_link_libraries(rt_unit_tests2 PRIVATE bitsy_rt)
    if(BT_COMPACT_VULKAN)
      target_compile_definitions(rt_unit_tests PRIVATE RT_WITH_VULKAN)
      target_compile_definitions(rt_unit_tests2 PRIVATE RT_WITH_VULKAN)
    endif()
    add_test(NAME unit_compact_core COMMAND rt_unit_tests)
    add_test(NAME unit_compact_core2 COMMAND rt_unit_tests2)
  endif()
endif()
