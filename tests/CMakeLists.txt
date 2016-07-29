link_libraries(ponos)

find_package(CppUnit REQUIRED)
include_directories(${CPPUNIT_INCLUDE_DIR})

include_directories("${PONOS_SOURCE_DIR}/src")

set(TESTS
TestBBox.cpp
TestMatrix.cpp
TestTransform.cpp
)

add_executable(run_tests main.cpp ${TESTS})
target_link_libraries(run_tests cppunit)
