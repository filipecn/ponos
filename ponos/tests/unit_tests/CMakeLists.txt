set(target ponos_unit_tests)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${target} ${SOURCES})
if(WIN32)
target_link_libraries(${target} ponos gtest)
else(WIN32)
target_link_libraries(${target} ponos gtest gcov pthread)
endif(WIN32)
