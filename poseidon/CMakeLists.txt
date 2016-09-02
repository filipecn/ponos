project(POSEIDON)

set(HEADERS
	src/elements/particle.h
	src/math/conjugate_gradient.h
	src/structures/mac_grid.h
	src/structures/particle_grid.h
	src/structures/particle_set.h
	src/structures/vdb_grid.h
	src/structures/vdb_mac_grid.h
	src/structures/vdb_particle_grid.h
	src/poseidon.h
	)

set(SOURCES
	src/elements/particle.cpp
	src/math/conjugate_gradient.cpp
	src/structures/particle_grid.cpp
	src/structures/vdb_grid.cpp
	src/structures/vdb_mac_grid.cpp
	src/structures/vdb_particle_grid.cpp
	)

link_libraries(ponos)
include_directories("${PONOS_SOURCE_DIR}/src")
include_directories("${POSEIDON_SOURCE_DIR}/src")
include_directories("${POSEIDON_SOURCE_DIR}/external")
include_directories("/home/filipecn/gitclones/OpenVDB/include")

add_library(poseidon ${SOURCES} ${HEADERS})
set_target_properties(poseidon PROPERTIES
	OUTPUT_NAME "poseidon"
	FOLDER "POSEIDON")

target_link_libraries(poseidon ${poseidon_LIBRARIES} -ltbb
	/home/filipecn/gitclones/OpenVDB/lib/libopenvdb.so
	/home/filipecn/gitclones/OpenVDB/lib/libopenvdb_points.so)
add_subdirectory(examples)
add_subdirectory(tests)
