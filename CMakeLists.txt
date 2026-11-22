cmake_minimum_required(VERSION 3.20)

project(bitsy VERSION 0.2.0 LANGUAGES C
        DESCRIPTION "Bitsy 0.2 — compact C+Vulkan lazy tensor runtime")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(BT_BUILD_TESTS "Build the C unit tests" OFF)
option(BT_COMPACT_VULKAN "Enable Vulkan in the compact runtime" OFF)
# Kept as a no-op for existing CI/config commands; the compact runtime is now
# the only runtime, so it is always built (PR #11 legacy cutover).
option(BT_BUILD_COMPACT "Compatibility no-op (compact runtime always built)" ON)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# v0.2 compact core: the only runtime after the legacy cutover (PR #11).
add_library(bitsy_rt SHARED src/rt/rt.c src/rt/rt_vk.c src/rt/rt_cudanative.c)
target_include_directories(bitsy_rt PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/bitsy_rt)
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
  set(RT_SHADERS fused fused_sp gemm gemm_vec4 reduce softmax layernorm transpose sgd)
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

if(BT_BUILD_TESTS)
  enable_testing()
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
