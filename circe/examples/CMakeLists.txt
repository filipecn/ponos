#link_libraries(circe ${CIRCE_LIBRARIES}
#${PONOS_LIBRARIES})
#include_directories(${PONOS_SOURCE_DIR} ${CIRCE_SOURCE_DIR} ${GLAD_INCLUDES} ${EIGEN_INCLUDES})
#add_executable(hello_circe_test WIN32 hello_circe.cpp)
#add_dependencies(hello_circe_test circe)
#add_executable(volume_box_example WIN32 volume_box_example.cpp)
#add_dependencies(volume_box_example circe)
#add_executable(volume_box2_example WIN32 volume_box2_example.cpp)
#add_dependencies(volume_box2_example circe)
#add_executable(compute_shader WIN32 compute_shader.cpp)
#add_executable(save_viewport_example save_viewport_example.cpp)
#target_include_directories(save_viewport_example PUBLIC ${LODEPNG_INCLUDES} ${ROOT_PROJECT_PATH}/external)
#target_link_libraries(save_viewport_example ${LODEPNG_LIBS})
#add_executable(viewports WIN32 viewports_example.cpp)
#add_executable(text_example text_example.cpp)
#target_compile_definitions(text_example PUBLIC -DASSETS_PATH="${CIRCE_SOURCE_DIR}/examples/assets" -DSHADERS_PATH="${CIRCE_SOURCE_DIR}/examples/shaders")
#add_executable(instances WIN32 instances.cpp)
#add_dependencies(instances circe)
#add_executable(scene_mesh WIN32 scene_mesh_example.cpp)
#add_executable(post_effects_example WIN32 post_effects_example.cpp)
#
#if (MSVC)
#        set(WINDOWS_BINARIES
#                hello_circe_test
##                bvh_example
#                #scene_app
#                compute_shader
# #               hemesh_example
#                save_viewport_example
#                viewports
#                text_example
#                instances
#                post_effects_example
#                volume_box_example
#                volume_box2_example
#        )
#    # Tell MSVC to use main instead of WinMain for Windows subsystem executables
#    set_target_properties(${WINDOWS_BINARIES} PROPERTIES
#            LINK_FLAGS "/ENTRY:mainCRTStartup")
#endif ()

# add_subdirectory(toratti)
# add_subdirectory(spheres)
 add_subdirectory(gl)
