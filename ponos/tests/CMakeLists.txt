set(SOURCES 
    main.cpp
    common_tests.cpp
    geometry_tests.cpp
    numeric_tests.cpp
    storage_tests.cpp
)

add_executable(ponos_tests ${SOURCES})
target_include_directories(ponos_tests PUBLIC ${CATCH2_INCLUDES})
target_link_libraries(ponos_tests ponos gcov)
