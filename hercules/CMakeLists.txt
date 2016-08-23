project(HERCULES)

set(HEADERS
	src/shapes/box.h
	src/body.hpp
	src/fixture.hpp
	src/rigid_body.h
	src/shape.hpp
	src/world.hpp
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
