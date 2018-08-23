set(target poseidon_unit_tests)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${target} ${SOURCES})
if(WIN32)
target_link_libraries(${target} poseidon gtest)
endif(WIN32)
if(APPLE)
	target_link_libraries(${target} ${PONOS_LIBRARIES} poseidon gtest pthread)
else(APPLE)
	target_link_libraries(${target} ${PONOS_LIBRARIES} poseidon gtest gcov pthread)
endif(APPLE)
