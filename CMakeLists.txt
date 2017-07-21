project(PONOS)

cmake_minimum_required(VERSION 2.8.7)

if(CMAKE_COMPILER_IS_GNUCXX)
  message(STATUS "GCC detected, adding compile flags")
	set(DEBUG_FLAGS     "${DEBUG_FLAGS} -std=c++11 -g -O3 -Wall -pedantic -Wextra -Werror -Wno-unused-parameter -Wno-unused-local-typedefs -fprofile-arcs -Wno-unused-result -ftest-coverage --coverage -Wno-conversion-null -Wno-unused-but-set-variable -DANSI_DECLARATORS -DTRILIBRARY -fPIC")
	set(RELEASE_FLAGS "${RELEASE_FLAGS} -std=c++11 -g -O3 -Wall -pedantic -Wextra -Werror -Wno-unused-parameter -Wno-unused-local-typedefs -fprofile-arcs -Wno-unused-result -ftest-coverage --coverage -Wno-conversion-null -Wno-unused-but-set-variable -DANSI_DECLARATORS -DTRILIBRARY -fPIC")
else(CMAKE_COMPILER_IS_GNUCXX)
	set(DEBUG_FLAGS   "/Zi /EHsc /MD /MDd /MP")
	set(RELEASE_FLAGS   "/EHsc")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake/modules")
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(CMAKE_CXX_FLAGS ${RELEASE_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG ${DEBUG_FLAGS})
set(CMAKE_CONFIGURATION_TYPES Debug Release)

include("${PROJECT_SOURCE_DIR}/cmake_dependencies.txt")
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(external/googletest EXCLUDE_FROM_ALL)
include_directories(external/googletest/googletest/include)

add_subdirectory(ponos)
if(NOT ONLYTESTS)
	add_subdirectory(aergia)
endif(NOT ONLYTESTS)
#add_subdirectory(hercules)
#add_subdirectory(odysseus)
#add_subdirectory(helios)

if(NOT TRAVIS)
	#add_subdirectory(poseidon)
endif(NOT TRAVIS)
