project(AERGIA)

cmake_minimum_required(VERSION 2.8.7)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake/modules")
set(DEBUG_FLAGS "-std=c++11 -g -O1 -Wall -Wextra -Werror -pedantic -Wno-unused-variable")
set(RELEASE_FLAGS "-std=c++11 -O3 -Wall -Wextra -Werror -pedantic -Wno-unused-variable")

if (WIN32)
  find_library(GLFW_LIB NAMES glfw3 PATHS ${PROJECT_SOURCE_DIR}/external/GLFW)
  include_directories("${PROJECT_SOURCE_DIR}/external")
  #find_library(VULKAN_LIB NAMES vulkan-1 vulkan)
  find_package(Vulkan REQUIRED)
else(WIN32)
	find_package(glfw3 REQUIRED)
	include_directories(${GLFW_INCLUDE_DIRS})
	find_package(OpenGL REQUIRED)
	include_directories(${OPENGL_INCLUDE_DIRS})
endif(WIN32)

link_libraries(ponos)
include_directories("${PONOS_SOURCE_DIR}/src")

if(WIN32)
	link_libraries(${GLFW_LIB} ${VULKAN_LIBRARY} opengl32 glu32)
else(WIN32)
	link_libraries(glfw ${OPENGL_LIBRARIES})
endif(WIN32)

add_subdirectory(src)
add_subdirectory(examples)
