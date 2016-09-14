project(PONOS)

cmake_minimum_required(VERSION 2.8.7)

set(DEBUG_FLAGS "-std=c++11 -g -O1 -Wall -Wextra -Werror -pedantic -Wno-unused-parameter -fprofile-arcs -ftest-coverage --coverage")
set(RELEASE_FLAGS "-std=c++11 -g -O3 -Wall -Wextra -Werror -pedantic -Wno-unused-parameter -fprofile-arcs -ftest-coverage --coverage")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

set(CMAKE_CXX_FLAGS ${RELEASE_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG ${DEBUG_FLAGS})
set(CMAKE_CONFIGURATION_TYPES Debug Release)


add_subdirectory(ponos)
if(NOT TRAVIS)
	add_subdirectory(aergia)
	add_subdirectory(poseidon)
	add_subdirectory(hercules)
	add_subdirectory(odysseus)
	add_subdirectory(helios)
endif(TRAVIS)

