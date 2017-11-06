project(PONOS)

cmake_minimum_required(VERSION 3.1)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE "ON")

if(CMAKE_COMPILER_IS_GNUCXX)
	set(DEBUG_FLAGS     "${DEBUG_FLAGS} -std=c++11 -g -Wall -Wextra -Werror --coverage")
	set(RELEASE_FLAGS "-std=c++11 -O3 -Wall -Wextra -Werror --coverage -fPIC")
endif(CMAKE_COMPILER_IS_GNUCXX)
if(MINGW)
    set(RELEASE_FLAGS "-std=c++11 -O3 -Wall -Wextra -Werror --coverage")
endif(MINGW)
if(MSVC)
  set(DEBUG_FLAGS   "/Zi /EHsc /MD /MDd")
  set(RELEASE_FLAGS   "/EHsc")
endif(MSVC)

set(ROOT_PROJECT_PATH "${PROJECT_SOURCE_DIR}")

set(CMAKE_CXX_FLAGS ${RELEASE_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG ${DEBUG_FLAGS})
set(CMAKE_CONFIGURATION_TYPES Debug Release)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(external/googletest EXCLUDE_FROM_ALL)
include_directories(external/googletest/googletest/include)

set(TRIANGLE_INCLUDE "")
include("${PROJECT_SOURCE_DIR}/CMakeDependencies.txt")
if (NOT("${TRIANGLE_INCLUDE}" STREQUAL "" OR "${TRIANGLE_LIB}" STREQUAL ""))
	add_definitions(-DTRIANGLE_INCLUDED)
endif (NOT("${TRIANGLE_INCLUDE}" STREQUAL "" OR "${TRIANGLE_LIB}" STREQUAL ""))

add_subdirectory(ponos)
# add_subdirectory(hercules)
# add_subdirectory(odysseus)
# add_subdirectory(helios)

if(NOT TRAVIS)
    add_subdirectory(aergia)
#	add_subdirectory(poseidon)
endif(NOT TRAVIS)
