project(POSEIDON)

set(POSEIDON_HEADERS
        poseidon/poseidon.h
        poseidon/numeric/staggered_grid.h
        poseidon/numeric/vector_fields.h
        poseidon/numeric/vector_grid.h
        )

set(POSEIDON_SOURCES
        poseidon/numeric/staggered_grid.cpp
        poseidon/numeric/vector_fields.cpp
        poseidon/numeric/vector_grid.cpp
        )

add_library(poseidon ${POSEIDON_SOURCES} ${POSEIDON_HEADERS})
set_target_properties(poseidon PROPERTIES
        OUTPUT_NAME "poseidon"
        FOLDER "POSEIDON")
target_include_directories(poseidon PUBLIC
        ${PONOS_SOURCE_DIR}
        ${HERMES_SOURCE_DIR}
        ${POSEIDON_SOURCE_DIR})
target_link_libraries(poseidon PUBLIC ponos hermes)

if (BUILD_EXAMPLES)
    add_subdirectory(examples/shallow_waters)
endif (BUILD_EXAMPLES)
if (BUILD_TESTS)
    add_subdirectory(tests/cuda_unit_tests)
    add_subdirectory(tests/unit_tests)
endif (BUILD_TESTS)

