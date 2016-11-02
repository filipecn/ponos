link_libraries(aergia)

link_libraries(${GLFW_LIB} ${GLEW_LIB} ${aergia_LIBRARIES} ${OPENGL_LIBRARIES})

include_directories("${AERGIA_SOURCE_DIR}/src")

add_executable(hello_aergia_test WIN32 hello_aergia.cpp)
add_executable(scene_app WIN32 scene_app.cpp)
add_executable(hello_3d WIN32 hello3D.cpp)
add_executable(hello_vulkan WIN32 hello_vulkan.cpp)
set(WINDOWS_BINARIES hello_aergia_test scene_app hello_3d hello_vulkan)

if (MSVC)
	# Tell MSVC to use main instead of WinMain for Windows subsystem executables
	set_target_properties(${WINDOWS_BINARIES} PROPERTIES
		LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()
