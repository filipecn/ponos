project(POSEIDON)

set(HEADERS
	src/algorithms/fast_sweep.h
	src/algorithms/solver.h
  src/elements/particle.h
  src/math/conjugate_gradient.h
  src/structures/mac_grid.h
  src/structures/particle_grid.h
  src/structures/particle_set.h
  src/structures/z_particle_grid.h
  src/poseidon.h
)

set(SOURCES
	src/algorithms/fast_sweep.cpp
  src/elements/particle.cpp
  src/math/conjugate_gradient.cpp
  src/structures/particle_grid.cpp
)

if(NOT WIN32 AND NOT NOOPENVDB)
  set(HEADERS ${HEADERS}
    src/structures/vdb_grid.h
    src/structures/vdb_level_set.h
    src/structures/vdb_mac_grid.h
		#    src/structures/vdb_particle_grid.h
  )
  set(SOURCES ${SOURCES}
    src/structures/vdb_grid.cpp
    src/structures/vdb_level_set.cpp
    src/structures/vdb_mac_grid.cpp
		#    src/structures/vdb_particle_grid.cpp
  )
endif(NOT WIN32 AND NOT NOOPENVDB)

link_libraries(ponos)
include_directories("${PONOS_SOURCE_DIR}/src")
link_libraries(aergia)
include_directories("${AERGIA_SOURCE_DIR}/src")
include_directories("${POSEIDON_SOURCE_DIR}/src")
include_directories("${POSEIDON_SOURCE_DIR}/external")
include_directories("C:/Users/fuiri/Downloads")
include_directories("/usr/local/include/eigen3")
include_directories("C:/Users/fuiri/Desktop/eigen-eigen-67e894c6cd8f")
if(NOT WIN32)
	#  include_directories("/home/filipecn/gitclones/OpenVDB/include")
  include_directories("/usr/local/OpenVDB/4.0.0/include")
	include_directories("/home/filipecn/gits/tbb2017_20161128oss/include")
endif(NOT WIN32)
add_library(poseidon ${SOURCES} ${HEADERS})
set_target_properties(poseidon PROPERTIES
  OUTPUT_NAME "poseidon"
FOLDER "POSEIDON")

if(WIN32 OR NOOPENVDB)
target_link_libraries(poseidon ${poseidon_LIBRARIES})
else(WIN32)
	add_definitions(-DOPENVDB_3_ABI_COMPATIBLE)
	target_link_libraries(poseidon ${poseidon_LIBRARIES}
	#-ltbb
	/home/filipecn/gits/tbb2017_20161128oss/build/linux_intel64_gcc_cc5.4.1_libc2.19_kernel3.4.0+_release/libtbb.so
	#/usr/local/OpenVDB/4.0.0/lib/libopenvdb.a
	/usr/local/OpenVDB/4.0.0/lib/libopenvdb.so
	#/usr/local/OpenVDB/4.0.0/lib/libopenvdb.so.4.0.1
	#/usr/local/OpenVDB/4.0.0/lib/libopenvdb.so.4.0.0
	#  /home/filipecn/gitclones/OpenVDB/lib/libopenvdb.so
	#  /home/filipecn/gitclones/OpenVDB/lib/libopenvdb_points.so
	)
endif(WIN32 OR NOOPENVDB)

add_subdirectory(examples)

if(TESTPOSEIDON)
  add_subdirectory(tests)
endif(TESTPOSEIDON)
