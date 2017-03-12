link_libraries(aergia)

link_libraries(${GLFW_LIB} ${GLEW_LIB} ${aergia_LIBRARIES} ${OPENGL_LIBRARIES})

include_directories("${AERGIA_SOURCE_DIR}/src")

add_executable(spheres_ia WIN32 spheres_ia.cpp)

if (MSVC)
	# Tell MSVC to use main instead of WinMain for Windows subsystem executables
	set_target_properties(spheres_ia PROPERTIES
		LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()
