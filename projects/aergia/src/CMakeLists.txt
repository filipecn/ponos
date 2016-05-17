include_directories("${AERGIA_SOURCE_DIR}/src")

set(HEADERS
io/graphics_display.h
aergia.h
)

set(SOURCES
io/graphics_display.cpp
)

add_library(aergia ${SOURCES} ${HEADERS})
set_target_properties(aergia PROPERTIES
                      OUTPUT_NAME "aergia"
                    FOLDER "AERGIA")

if (WIN32)
    # The GLFW DLL needs a special compile-time macro and import library name
    #set_target_properties(glfw PROPERTIES PREFIX "" IMPORT_PREFIX "")
    if (MINGW)
        set_target_properties(aergia PROPERTIES IMPORT_SUFFIX "dll.a")
    else()
        set_target_properties(aergia PROPERTIES IMPORT_SUFFIX "dll.lib")
    endif()
endif()

target_link_libraries(aergia ${aergia_LIBRARIES})
