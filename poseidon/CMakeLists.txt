project(POSEIDON)

set(HEADERS
        poseidon/structures/particle_system.h
        poseidon/structures/particle_system.inl
        poseidon/math/numerical.h
        poseidon/poseidon.h
        poseidon/helpers/particle_system_model.h
        )

set(SOURCES
        poseidon/structures/particle_system.cpp
        poseidon/math/numerical.cpp
        poseidon/helpers/particle_system_model.cpp
        )

#if(NOT WIN32 AND NOT NOOPENVDB)
#  set(HEADERS ${HEADERS}
#    helios/structures/vdb_grid.h
#    helios/structures/vdb_level_set.h
#    helios/structures/vdb_mac_grid.h
#		#    helios/structures/vdb_particle_grid.h
#  )
#  set(SOURCES ${SOURCES}
#    helios/structures/vdb_grid.cpp
#    helios/structures/vdb_level_set.cpp
#    helios/structures/vdb_mac_grid.cpp
#		#    helios/structures/vdb_particle_grid.cpp
#  )
#endif(NOT WIN32 AND NOT NOOPENVDB)

link_libraries(${AERGIA_LIBRARIES})
include_directories(${AERGIA_INCLUDES})
include_directories(${POSEIDON_INCLUDES})
link_libraries(${POSEIDON_LIBRARIES})

link_libraries(ponos ${PONOS_LIBRARIES})
include_directories("${PONOS_SOURCE_DIR}")
link_libraries(aergia)
include_directories("${AERGIA_SOURCE_DIR}")

include_directories("${POSEIDON_SOURCE_DIR}")

add_library(poseidon ${SOURCES} ${HEADERS})
set_target_properties(poseidon PROPERTIES
        OUTPUT_NAME "poseidon"
        FOLDER "POSEIDON")

add_subdirectory(examples)

if (TESTPOSEIDON)
    #	add_subdirectory(tests)
endif (TESTPOSEIDON)

add_subdirectory(tests/unit_tests)
