project(POSEIDON)

set(HEADERS
	src/algorithms/solver.h
	src/algorithms/grid_solver.h
	src/elements/collider.h
  src/elements/particle.h
	src/elements/simulation_scene.h
  # src/math/conjugate_gradient.h
  src/structures/mac_grid.h
  src/structures/particle_grid.h
  src/structures/particle_set.h
  src/structures/z_particle_grid.h
  src/poseidon.h
)

set(SOURCES
	src/algorithms/grid_solver.cpp
  src/elements/particle.cpp
	src/elements/simulation_scene.cpp
  # src/math/conjugate_gradient.cpp
  src/structures/particle_grid.cpp
)

#if(NOT WIN32 AND NOT NOOPENVDB)
#  set(HEADERS ${HEADERS}
#    src/structures/vdb_grid.h
#    src/structures/vdb_level_set.h
#    src/structures/vdb_mac_grid.h
#		#    src/structures/vdb_particle_grid.h
#  )
#  set(SOURCES ${SOURCES}
#    src/structures/vdb_grid.cpp
#    src/structures/vdb_level_set.cpp
#    src/structures/vdb_mac_grid.cpp
#		#    src/structures/vdb_particle_grid.cpp
#  )
#endif(NOT WIN32 AND NOT NOOPENVDB)

link_libraries(${AERGIA_LIBRARIES})
include_directories(${AERGIA_INCLUDES})
include_directories(${POSEIDON_INCLUDES})
link_libraries(${POSEIDON_LIBRARIES})

link_libraries(ponos)
include_directories("${PONOS_SOURCE_DIR}/src")
link_libraries(aergia)
include_directories("${AERGIA_SOURCE_DIR}/src")

include_directories("${POSEIDON_SOURCE_DIR}/src")

add_library(poseidon ${SOURCES} ${HEADERS})
set_target_properties(poseidon PROPERTIES
  OUTPUT_NAME "poseidon"
FOLDER "POSEIDON")

add_subdirectory(examples)

if(TESTPOSEIDON)
	add_subdirectory(tests)
endif(TESTPOSEIDON)

add_subdirectory(tests/unit_tests)
