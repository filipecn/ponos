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

add_custom_target(tests
	COMMAND ponos_unit_tests DEPENDS ponos_unit_tests
	COMMAND lcov --directory .. --capture -output-file cov.info
	COMMAND lcov --remove cov.info '/usr/*' --output-file cov.info
	COMMAND lcov --remove cov.info '*/external/*' --output-file cov.info.cleaned
	COMMAND genhtml cov.info.cleaned --output-directory out
	)
