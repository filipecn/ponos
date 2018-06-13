project(PONOS)

cmake_minimum_required(VERSION 3.1)
# Google Tests
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(external/googletest EXCLUDE_FROM_ALL)
set(GOOGLE_TEST_INCLUDES "external/googletest/googletest/include")
# GLFW
#set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
#set(GLFW_BUILD_TESTS OFF CACHE BOOL " " FORCE)
#set(GLFW_BUILD_DOCS OFF CACHE BOOL " " FORCE)
#set(GLFW_BUILD_INSTALL OFF CACHE BOOL " " FORCE)
#set(GLFW_INSTALL OFF CACHE BOOL " " FORCE)
#set(GLFW_USE_CHDIR OFF CACHE BOOL " " FORCE)
#set(BUILD_SHARED_LIBS ${PONOS_BUILD_SHARED} CACHE BOOL " " FORCE)
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/glfw" "external_build/glfw")
#set_target_properties(glfw PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
# nanogui
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/nanogui" "external_build/nanogui")
set_target_properties(nanogui PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
set(NANOGUI_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/external/nanogui/include")
set(EIGEN_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/external/nanogui/ext/eigen")
set(NANOVG_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/external/nanogui/ext/nanovg/src")
set(NANOGUI_LIBS glfw GL Xxf86vm Xrandr Xinerama Xcursor Xi X11 pthread nanogui)
# tiny obj
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobj" "external_build/tinyobj")
set_target_properties(tinyobjloader PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
set(TINYOBJLOADER_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobj")
set(TINYOBJLOADER_LIBS tinyobjloader)
# Triangle
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/triangle" "external_build/triangle")
set_target_properties(triangle PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
set(TRIANGLE_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/external/triangle")
set(TRIANGLE_LIBS triangle)
# stb image library
set(STB_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/external")
# ply reader
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/ply" "external_build/ply")
set_target_properties(ply PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
set(PLY_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/external/ply")
set(PLY_LIBS ply)
# glad
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/glad" "external_build/glad")
set_target_properties(glad PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
target_compile_definitions(glad PUBLIC GLAD_GENERATOR="c++")
set(GLAD_INCLUDES "${CMAKE_BINARY_DIR}/external_build/glad/include")

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE "ON")

if (CMAKE_COMPILER_IS_GNUCXX)
    set(DEBUG_FLAGS "${DEBUG_FLAGS} -std=c++11 -g -Wall -Wextra --coverage")
    set(RELEASE_FLAGS "-std=c++11 -O3 -Wall -Wextra --coverage -fPIC")
endif (CMAKE_COMPILER_IS_GNUCXX)
if (MINGW)
    set(RELEASE_FLAGS "-std=c++11 -O3 -Wall -Wextra -Werror --coverage")
endif (MINGW)
if (MSVC)
    set(DEBUG_FLAGS "/Zi /EHsc /MD /MDd")
    set(RELEASE_FLAGS "/EHsc")
endif (MSVC)

set(ROOT_PROJECT_PATH "${PROJECT_SOURCE_DIR}")

set(CMAKE_CXX_FLAGS ${RELEASE_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG ${DEBUG_FLAGS})
set(CMAKE_CONFIGURATION_TYPES Debug Release)
set(TRIANGLE_INCLUDE "")
set(TETGEN_INCLUDE "")

# include("${PROJECT_SOURCE_DIR}/CMakeDependencies.txt")

if (NOT ("${TRIANGLE_INCLUDE}" STREQUAL "" OR "${TRIANGLE_LIB}" STREQUAL ""))
    add_definitions(-DTRIANGLE_INCLUDED)
endif (NOT ("${TRIANGLE_INCLUDE}" STREQUAL "" OR "${TRIANGLE_LIB}" STREQUAL ""))
if (NOT ("${TETGEN_INCLUDE}" STREQUAL "" OR "${TETGEN_LIB}" STREQUAL ""))
    add_definitions(-DTETGEN_INCLUDED)
endif (NOT ("${TETGEN_INCLUDE}" STREQUAL "" OR "${TETGEN_LIB}" STREQUAL ""))

add_subdirectory(ponos)
# add_subdirectory(hercules)
# add_subdirectory(odysseus)
# add_subdirectory(helios)

if (NOT TRAVIS)
    add_subdirectory(aergia)
    add_subdirectory(poseidon)
endif (NOT TRAVIS)
