include_directories("${HERCULES_SOURCE_DIR}/src")

set(HEADERS
	body.hpp
	fixture.hpp
	shape.hpp
	world.hpp
	hercules.h
	)

set(SOURCES
	)

add_library(hercules ${SOURCES} ${HEADERS})
set_target_properties(hercules PROPERTIES
	OUTPUT_NAME "hercules"
	FOLDER "HERCULES")

if (WIN32)
	# The GLFW DLL needs a special compile-time macro and import library name
	#set_target_properties(glfw PROPERTIES PREFIX "" IMPORT_PREFIX "")
	if (MINGW)
		set_target_properties(aergia PROPERTIES IMPORT_SUFFIX "dll.a")
	else()
		set_target_properties(aergia PROPERTIES IMPORT_SUFFIX "dll.lib")
	endif()
endif()

target_link_libraries(hercules ${hercules_LIBRARIES})
