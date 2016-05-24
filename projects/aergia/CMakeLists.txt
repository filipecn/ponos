project(AERGIA)

cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake/modules")

if (WIN32)
  find_library(GLFW_LIB NAMES glfw3 PATHS ${PROJECT_SOURCE_DIR}/external/GLFW)
  include_directories("${PROJECT_SOURCE_DIR}/external")
  #find_library(VULKAN_LIB NAMES vulkan-1 vulkan)
  find_package(Vulkan REQUIRED)
#  message(FATAL_ERROR "${VULKAN_LIB}")
endif()

link_libraries(ponos)
include_directories("${PONOS_SOURCE_DIR}/src")

add_subdirectory(src)
add_subdirectory(examples)
