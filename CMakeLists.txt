project(PONOS)

cmake_minimum_required(VERSION 2.8.7)

if(CMAKE_COMPILER_IS_GNUCXX)
  message(STATUS "GCC detected, adding compile flags")
	set(DEBUG_FLAGS     "${DEBUG_FLAGS} -std=c++11 -g -O3 -Wall -pedantic -Wextra -Werror -Wno-unused-parameter -Wno-unused-local-typedefs -fprofile-arcs -Wno-unused-result -ftest-coverage --coverage")
  set(RELEASE_FLAGS "${RELEASE_FLAGS} -std=c++11 -g -O3 -Wall -pedantic -Wextra -Werror -Wno-unused-parameter -Wno-unused-local-typedefs -fprofile-arcs -Wno-unused-result -ftest-coverage --coverage")
else(CMAKE_COMPILER_IS_GNUCXX)
  set(DEBUG_FLAGS   "/Zi /EHsc /MD /MDd")
  set(RELEASE_FLAGS "   ")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(CMAKE_CXX_FLAGS ${RELEASE_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG ${DEBUG_FLAGS})
set(CMAKE_CONFIGURATION_TYPES Debug Release)

add_subdirectory(ponos)
if(NOT ONLYTESTS)
add_subdirectory(aergia)
endif(NOT ONLYTESTS)
add_subdirectory(hercules)
#add_subdirectory(odysseus)
add_subdirectory(helios)

if(NOT TRAVIS)
	add_subdirectory(poseidon)
endif(NOT TRAVIS)
