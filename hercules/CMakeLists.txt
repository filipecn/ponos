project(HERCULES)

set(HEADERS
	src/2d/fixture.h
	src/2d/rigid_body.h
	src/2d/world.h
	src/cds/aabb_object_interface.h
	src/cds/aabb_grid.h
	src/cds/cds.h
	src/world_interface.h
	src/hercules.h
	)

set(SOURCES
	src/shapes/box.cpp
	)

link_libraries(ponos)
include_directories("${PONOS_SOURCE_DIR}/src")
include_directories("${HERCULES_SOURCE_DIR}/src")

add_library(hercules ${SOURCES} ${HEADERS})
set_target_properties(hercules PROPERTIES
	OUTPUT_NAME "hercules"
	FOLDER "HERCULES")

target_link_libraries(hercules ${hercules_LIBRARIES})

# tests
add_subdirectory(tests)
