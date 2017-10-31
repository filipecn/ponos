link_libraries(ponos)
link_libraries(poseidon)

include_directories("${CPPUNIT_INCLUDE_DIR}")
link_libraries(${CPPUNIT_LIB})

set(TESTS
		TestFLIP2D.cpp
		TestZParticleGrid.cpp
		TestSweep.cpp
)

if(NOT NOOPENVDB)
	include_directories("${POSEIDON_SOURCE_DIR}/examples/openvdb")
	include_directories("${POSEIDON_SOURCE_DIR}/examples/flip2d")
	set(TESTS
		TestFLIP2D.cpp
		TestZParticleGrid.cpp
		TestSweep.cpp
		TestFLIP.cpp
		TestVDBGrid.cpp
		TestVDBMacGrid.cpp
		TestVDBParticleGrid.cpp
		)
	add_executable(run_poseidon_tests main.cpp
		# ${POSEIDON_SOURCE_DIR}/examples/openvdb/flip.cpp
	  ${POSEIDON_SOURCE_DIR}/examples/flip2d/flip_2d.h
		${TESTS})
endif(NOT NOOPENVDB)

if(NOT NOOPENVDB)
	target_link_libraries(run_poseidon_tests cppunit gcov -ltbb -pthread
		${OPENVDB_LIB})
endif(NOT NOOPENVDB)
