project(PONOS)

cmake_minimum_required(VERSION 2.8.7)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake/modules")

set(DEBUG_FLAGS "-std=c++11 -g -O1 -Wall -Wextra -pedantic")
set(RELEASE_FLAGS "-std=c++11 -O3 -Wall -Wextra -pedantic")

set(CMAKE_CXX_FLAGS ${RELEASE_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG ${DEBUG_FLAGS})
set(CMAKE_CONFIGURATION_TYPES Debug Release)

add_subdirectory(src)
add_subdirectory(tests)
#add_subdirectory(projects)
