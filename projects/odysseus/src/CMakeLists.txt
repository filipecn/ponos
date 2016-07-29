include_directories("${ODYSSEUS_SOURCE_DIR}/src")

set(HEADERS
	game.hpp
	game_object.hpp
	graphics_component.hpp
	input_component.hpp
	physics_component.hpp
	clock.hpp
	odysseus.h
	)

set(SOURCES
	)

add_library(odysseus ${SOURCES} ${HEADERS})
set_target_properties(odysseus PROPERTIES
	OUTPUT_NAME "odysseus"
	FOLDER "ODYSSEUS")

if (WIN32)
	# The GLFW DLL needs a special compile-time macro and import library name
	#set_target_properties(glfw PROPERTIES PREFIX "" IMPORT_PREFIX "")
	if (MINGW)
		set_target_properties(aergia PROPERTIES IMPORT_SUFFIX "dll.a")
	else()
		set_target_properties(aergia PROPERTIES IMPORT_SUFFIX "dll.lib")
	endif()
endif()

target_link_libraries(odysseus ${odysseus_LIBRARIES})
