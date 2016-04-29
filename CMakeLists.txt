project(PONOS)

cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake/modules")

add_subdirectory(src)
add_subdirectory(tests)
