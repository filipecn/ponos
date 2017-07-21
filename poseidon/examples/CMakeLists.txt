find_library(LAPACK_LIB NAMES liblapack PATHS ${POSEIDON_SOURCE_DIR}/external)

link_libraries(${AERGIA_LIBRARIES})
include_directories(${AERGIA_INCLUDES})
link_libraries( -pthread ${POSEIDON_SOURCE_DIR}/external/libcblas.dll.a)
#include_directories("${FREETYPE_INCLUDE}")
#link_libraries(${FREETYPE_LIBRARIES})
link_libraries(poseidon)
include_directories("${POSEIDON_SOURCE_DIR}/src")
include_directories("${POSEIDON_SOURCE_DIR}/external")
link_libraries(ponos)
include_directories("${PONOS_SOURCE_DIR}/src")
link_libraries(aergia)
include_directories("${AERGIA_SOURCE_DIR}/src")

include_directories("${POSEIDON_SOURCE_DIR}/examples")

add_executable(hello_grid_solver WIN32 hello_grid_solver.cpp)

if(NOT NOGRAPHICS AND NOT NOOPENVDB)
	add_executable(zgrid WIN32 zgrid.cpp)

set(WINDOWS_BINARIES zgrid hello_grid_solver)

if (MSVC)
    # Tell MSVC to use main instead of WinMain for Windows subsystem executables
		set_target_properties(${WINDOWS_BINARIES} PROPERTIES
		                     LINK_FLAGS "/ENTRY:mainCRTStartup")
endif(MSVC)

# add_subdirectory(flip2d)

if(NOT WIN32)
	# add_subdirectory(openvdb)
endif(NOT WIN32)

endif(NOT NOGRAPHICS AND NOT NOOPENVDB)
