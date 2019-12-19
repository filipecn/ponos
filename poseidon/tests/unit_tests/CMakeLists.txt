set(target poseidon_unit_tests)

file(GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/numeric_tests.cpp
        )

add_executable(${target} ${SOURCES})
target_include_directories(${target} PUBLIC ${GOOGLE_TEST_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR})

if (WIN32)
    target_link_libraries(${target} poseidon gtest)
elseif (APPLE)
    target_link_libraries(${target} poseidon gtest pthread)
else (WIN32)
    target_link_libraries(${target} poseidon gtest gcov pthread)
endif (WIN32)

add_custom_target(test_poseidon
        COMMAND poseidon_unit_tests DEPENDS poseidon_unit_tests
        COMMAND lcov --directory .. --capture -output-file cov.info
        COMMAND lcov --remove cov.info '/usr/*' --output-file cov.info
        COMMAND lcov --remove cov.info '*/external/*' --output-file cov.info.cleaned
        COMMAND genhtml cov.info.cleaned --output-directory out
        )
