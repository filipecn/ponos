link_libraries(ponos)

include_directories("${PONOS_SOURCE_DIR}/src")

set(TESTS
	TestBBox.cpp
	TestQuaternion.cpp
	TestQueries.cpp
	TestMatrix.cpp
	TestObjectPool.cpp
	TestParallel.cpp
	TestSearch.cpp
	TestSort.cpp
	TestTransform.cpp
	)

if(WIN32)
include_directories("C:/Users/fuiri/Desktop/cppunit-1.13.2/include")
link_libraries("C:/Users/fuiri/Desktop/cppunit-1.13.2/lib/cppunit.lib")
add_executable(run_ponos_tests main.cpp ${TESTS})
target_link_libraries(run_ponos_tests)
else(WIN32)
find_package(CppUnit REQUIRED)
include_directories(${CPPUNIT_INCLUDE_DIR})
add_executable(run_ponos_tests main.cpp ${TESTS})
target_link_libraries(run_ponos_tests cppunit gcov pthread)
endif(WIN32)
