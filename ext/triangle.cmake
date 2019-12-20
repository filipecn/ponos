add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/ext/triangle" "deps_build/triangle")
set_target_properties(triangle PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
set(TRIANGLE_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/deps/triangle")
set(TRIANGLE_LIBS triangle)

set(TRIANGLE_INCLUDE "")
set(TETGEN_INCLUDE "")
if (NOT ("${TRIANGLE_INCLUDE}" STREQUAL "" OR "${TRIANGLE_LIB}" STREQUAL ""))
    add_definitions(-DTRIANGLE_INCLUDED)
endif (NOT ("${TRIANGLE_INCLUDE}" STREQUAL "" OR "${TRIANGLE_LIB}" STREQUAL ""))
if (NOT ("${TETGEN_INCLUDE}" STREQUAL "" OR "${TETGEN_LIB}" STREQUAL ""))
    add_definitions(-DTETGEN_INCLUDED)
endif (NOT ("${TETGEN_INCLUDE}" STREQUAL "" OR "${TETGEN_LIB}" STREQUAL ""))