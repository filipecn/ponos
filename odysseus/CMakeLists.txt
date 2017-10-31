project(ODYSSEUS)

set(HEADERS
	src/game.hpp
	src/game_object.hpp
	src/component_interface.h
	src/graphics_component.hpp
	src/graphics_manager.h
	src/input_component.hpp
	src/physics_component.hpp
	src/physics_manager.h
	src/clock.hpp
	src/odysseus.h
	)

set(SOURCES
	src/graphics_manager.cpp
	)

link_libraries(ponos)
include_directories("${PONOS_SOURCE_DIR}/src")
link_libraries(aergia)
include_directories("${AERGIA_SOURCE_DIR}/src")
link_libraries(hercules)
include_directories("${HERCULES_SOURCE_DIR}/src")
include_directories("${ODYSSEUS_SOURCE_DIR}/src")

add_library(odysseus ${SOURCES} ${HEADERS})
set_target_properties(odysseus PROPERTIES
	OUTPUT_NAME "odysseus"
	FOLDER "ODYSSEUS")

target_link_libraries(odysseus ${odysseus_LIBRARIES})

add_subdirectory(examples)
