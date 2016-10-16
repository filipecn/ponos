link_libraries(ponos)
link_libraries(hercules)

find_package(CppUnit REQUIRED)
include_directories(${CPPUNIT_INCLUDE_DIR})

include_directories("${PONOS_SOURCE_DIR}/src")
include_directories("${HERCULES_SOURCE_DIR}/src")

set(TESTS
	TestAABBGrid.cpp
	TestAABBSweep.cpp
	)

add_executable(run_hercules_tests main.cpp ${TESTS})
target_link_libraries(run_hercules_tests cppunit gcov pthread)
