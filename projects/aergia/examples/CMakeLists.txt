link_libraries(aergia)

link_libraries(glfw ${aergia_LIBRARIES} ${OPENGL_LIBRARIES})

include_directories("${AERGIA_SOURCE_DIR}/src")

add_executable(hello_aergia_test WIN32 hello_aergia.cpp)

set(WINDOWS_BINARIES hello_aergia_test)

if (MSVC)
    # Tell MSVC to use main instead of WinMain for Windows subsystem executables
    set_target_properties(${WINDOWS_BINARIES} PROPERTIES
                          LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()
