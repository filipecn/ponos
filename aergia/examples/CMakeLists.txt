link_libraries(aergia ${AERGIA_LIBRARIES} ${PONOS_LIBRARIES} ${NANOGUI_LIBS})
add_executable(hello_aergia_test WIN32 hello_aergia.cpp)
add_executable(nanogui_example WIN32 nanogui_example.cpp)
target_include_directories(nanogui_example PUBLIC ${NANOGUI_INCLUDES} ${EIGEN_INCLUDES} ${NANOVG_INCLUDES})
add_executable(scene_app WIN32 scene_app.cpp)
add_executable(compute_shader WIN32 compute_shader.cpp)
add_executable(stream_lines WIN32 stream_lines.cpp)
add_executable(vector_grid WIN32 vector_grid_example.cpp)
add_executable(geometry_queries WIN32 geometry_queries.cpp)
add_executable(procedural_texture WIN32 procedural_texture.cpp)
add_executable(hemesh_example WIN32 hemesh.cpp)
add_executable(viewports WIN32 viewports_example.cpp)
add_executable(text_example text_example.cpp)
add_executable(instances WIN32 instances.cpp)
add_executable(scene_mesh WIN32 scene_mesh_example.cpp)
add_executable(post_effects_example WIN32 post_effects_example.cpp)
add_executable(tmesh WIN32 tmesh.cpp)
#add_executable(triangulate_example WIN32 triangulate_example.cpp)

if (WIN32)
    # add_executable(hello_vulkan WIN32 hello_vulkan.cpp)
endif (WIN32)

set(WINDOWS_BINARIES
        hemesh_example
        c_regular_grid_example
        compute_shader
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
endif ()

# add_subdirectory(toratti)
# add_subdirectory(spheres)
