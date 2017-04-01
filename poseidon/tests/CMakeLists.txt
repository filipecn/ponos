link_libraries(ponos)
link_libraries(poseidon)

find_package(CppUnit REQUIRED)
include_directories(${CPPUNIT_INCLUDE_DIR})

include_directories("${PONOS_SOURCE_DIR}/src")
include_directories("${POSEIDON_SOURCE_DIR}/src")

set(TESTS
		TestFLIP2D.cpp
		TestZParticleGrid.cpp
		TestSweep.cpp
)

if(NOT NOOPENVDB)
	include_directories("/home/filipecn/gitclones/OpenVDB/include")
include_directories("/usr/local/OpenVDB/4.0.0/include")
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
		/usr/local/OpenVDB/4.0.0/lib/libopenvdb.so)
		#		/home/filipecn/gitclones/OpenVDB/lib/libopenvdb.so
		#		/home/filipecn/gitclones/OpenVDB/lib/libopenvdb_points.so -lHalf)
endif(NOT NOOPENVDB)
