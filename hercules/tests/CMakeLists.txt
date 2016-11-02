link_libraries(ponos)
link_libraries(hercules)

include_directories("${PONOS_SOURCE_DIR}/src")
include_directories("${HERCULES_SOURCE_DIR}/src")

set(TESTS
	TestAABBGrid.cpp
	TestAABBSweep.cpp
	)


if(WIN32)
include_directories("C:/Users/fuiri/Desktop/cppunit-1.13.2/include")
link_libraries("C:/Users/fuiri/Desktop/cppunit-1.13.2/lib/cppunit.lib")
add_executable(run_hercules_tests main.cpp ${TESTS})
target_link_libraries(run_hercules_tests)
else(WIN32)
find_package(CppUnit REQUIRED)
include_directories(${CPPUNIT_INCLUDE_DIR})
add_executable(run_hercules_tests main.cpp ${TESTS})
target_link_libraries(run_hercules_tests cppunit gcov pthread)
endif(WIN32)
