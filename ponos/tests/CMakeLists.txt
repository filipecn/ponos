link_libraries(ponos)

set(TESTS
	TestBBox.cpp
	TestBRep.cpp
	TestQuaternion.cpp
	TestQueries.cpp
	TestMatrix.cpp
	TestObjectPool.cpp
	TestParallel.cpp
	TestSearch.cpp
	TestSort.cpp
	TestTransform.cpp
	TestZGrid.cpp
	)

add_executable(run_ponos_tests main.cpp ${TESTS})
target_link_libraries(run_ponos_tests cppunit gcov pthread)
