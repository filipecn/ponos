set(SOURCES
        main.cpp
        blas_tests.cpp
        common_tests.cpp
        geometry_tests.cpp
        log_tests.cpp
        numeric_tests.cpp
        storage_tests.cpp
        random_tests.cpp
        structures_tests.cpp
        parallel_tests.cpp
        )

add_executable(ponos_tests ${SOURCES})
target_include_directories(ponos_tests PUBLIC ${CATCH2_INCLUDES})
target_link_options(ponos_tests INTERFACE --coverage)

if (UNIX AND NOT APPLE)
    target_link_libraries(ponos_tests ponos gcov)

    add_custom_target(test_ponos
            COMMAND lcov --directory . --zerocounters
            COMMAND ponos_tests DEPENDS ponos_tests
            COMMAND lcov --directory . --capture -output-file cov.info
            COMMAND lcov --remove cov.info '/usr/*' --output-file cov.info
            COMMAND lcov --remove cov.info '*/ext/*' --output-file cov.info
            COMMAND lcov --remove cov.info '*/tests/*' --output-file cov.info
            COMMAND genhtml cov.info --output-directory out
            )
else (UNIX AND NOT APPLE)

    target_link_libraries(ponos_tests ponos)

endif (UNIX AND NOT APPLE)