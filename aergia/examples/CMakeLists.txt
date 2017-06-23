link_libraries(aergia ${AERGIA_LIBRARIES})

add_executable(hello_aergia_test WIN32 hello_aergia.cpp)
add_executable(scene_app WIN32 scene_app.cpp)
add_executable(hello_3d WIN32 hello3D.cpp)
add_executable(stream_lines WIN32 stream_lines.cpp)
add_executable(vector_grid WIN32 vector_grid_example.cpp)
add_executable(geometry_queries WIN32 geometry_queries.cpp)
add_executable(quad_tree WIN32 quad_tree.cpp)
add_executable(procedural_texture WIN32 procedural_texture.cpp)
add_executable(z_grid zgrid_example.cpp)
add_executable(c_regular_grid_example WIN32 c_regular_grid.cpp)
add_executable(hemesh_example WIN32 hemesh.cpp)

if(WIN32)
	# add_executable(hello_vulkan WIN32 hello_vulkan.cpp)
endif(WIN32)

set(WINDOWS_BINARIES
	hemesh_example
	c_regular_grid_example
  procedural_texture
	geometry_queries
	z_grid
	vector_grid
	hello_aergia_test
	scene_app
	hello_3d
	stream_lines
	quad_tree
)

if (MSVC)
	# Tell MSVC to use main instead of WinMain for Windows subsystem executables
	set_target_properties(${WINDOWS_BINARIES} PROPERTIES
		LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()

# add_subdirectory(spheres)
