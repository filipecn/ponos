link_libraries(aergia)

link_libraries(${AERGIA_LIBRARIES})

add_executable(toratti WIN32 toratti.cpp)

if (MSVC)
	# Tell MSVC to use main instead of WinMain for Windows subsystem executables
	set_target_properties(spheres_ia PROPERTIES
		LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()
