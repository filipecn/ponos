project(CIRCE)

set(HEADERS
        circe/colors/color.h
        circe/colors/color_palette.h
        circe/graphics/compute_shader.h
        circe/graphics/post_effect.h
        circe/graphics/shader.h
        circe/graphics/shader_manager.h
        circe/helpers/bvh_model.h
        circe/helpers/camera_model.h
        circe/helpers/cartesian_grid.h
        circe/helpers/geometry_drawers.h
        circe/helpers/grid_model.h
        circe/helpers/hemesh_model.h
        circe/helpers/quad_tree_model.h
        circe/helpers/scene_handle.h
        circe/helpers/vector_grid.h
        circe/helpers/tmesh_model.h
        circe/io/buffer.h
        circe/io/display_renderer.h
        circe/io/framebuffer.h
        circe/io/graphics_display.h
        circe/io/image_texture.h
        circe/io/font_texture.h
        circe/io/render_texture.h
        circe/io/screen_quad.h
        circe/io/storage_buffer.h
        circe/io/texture.h
        circe/io/texture_parameters.h
        circe/io/utils.h
        circe/io/viewport_display.h
        circe/io/user_input.h
        #  circe/io/vulkan_application.h
        # circe/io/vulkan_resource.h
        #  circe/io/vulkan_utils.h
        circe/scene/bvh.h
        circe/scene/camera.h
        # circe/scene/camera_2d.h
        circe/scene/camera_projection.h
        circe/scene/mesh_utils.h
        circe/scene/quad.h
        circe/scene/scene.h
        circe/scene/scene_mesh.h
        circe/scene/scene_object.h
        circe/scene/triangle_mesh.h
        circe/scene/wireframe_mesh.h
        circe/ui/app.h
        circe/ui/interactive_object_interface.h
        circe/ui/modifier_cursor.h
        # circe/ui/nanogui_screen.h
        circe/ui/scene_app.h
        circe/ui/text_renderer.h
        circe/ui/font_manager.h
        circe/ui/trackball.h
        circe/ui/trackball_interface.h
        circe/ui/track_mode.h
        circe/ui/ui_camera.h
        circe/utils/open_gl.h
        circe/utils/win32_utils.h
        circe/circe.h
        )

set(SOURCES
        circe/graphics/compute_shader.cpp
        circe/graphics/post_effect.cpp
        circe/graphics/shader.cpp
        circe/graphics/shader_manager.cpp
        circe/helpers/cartesian_grid.cpp
        circe/helpers/geometry_drawers.cpp
        circe/io/buffer.cpp
        circe/io/display_renderer.cpp
        circe/io/framebuffer.cpp
        circe/io/graphics_display.cpp
        circe/io/image_texture.cpp
        circe/io/render_texture.cpp
        circe/io/font_texture.cpp
        circe/io/storage_buffer.cpp
        circe/io/screen_quad.cpp
        circe/io/texture.cpp
        circe/io/utils.cpp
        circe/io/viewport_display.cpp
        #  circe/io/vulkan_application.cpp
        circe/scene/bvh.cpp
        circe/scene/camera.cpp
        # circe/scene/camera_2d.cpp
        circe/scene/mesh_utils.cpp
        circe/scene/quad.cpp
        circe/scene/scene_mesh.cpp
        circe/scene/triangle_mesh.cpp
        circe/scene/wireframe_mesh.cpp
        circe/ui/app.cpp
        # circe/ui/nanogui_screen.cpp
        circe/ui/text_renderer.cpp
        circe/ui/font_manager.cpp
        circe/ui/trackball_interface.cpp
        circe/ui/ui_camera.cpp
        circe/utils/open_gl.cpp
        circe/circe.cpp
        circe/scene/instance_set.cpp circe/scene/instance_set.h)

add_library(circe STATIC ${SOURCES} ${HEADERS})
set_target_properties(circe PROPERTIES
        OUTPUT_NAME "circe"
        FOLDER "CIRCE")

target_compile_definitions(circe PUBLIC -DASSET_PATH="${CIRCE_SOURCE_DIR}/examples/assets")
target_include_directories(circe PUBLIC ${CIRCE_SOURCE_DIR} ${STB_INCLUDES} ${PLY_INCLUDES} ${TINYOBJLOADER_INCLUDES} ${GLAD_INCLUDES} ${GLFW_INCLUDE_DIR})
target_link_libraries(circe PUBLIC ponos  ${GLFW_LIBRARIES} ${PLY_LIBS} ${TINYOBJLOADER_LIBS} glad)
#add_dependencies(circe glfw glad)

add_subdirectory(examples)

if (WIN32)
else (WIN32)
        SET(INSTALL_PATH ${ROOT_PROJECT_PATH}/build)
endif (WIN32)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/circe
        DESTINATION ${INSTALL_PATH}/include
        FILES_MATCHING REGEX "(.*\\.[inl|h])")

install(TARGETS circe #aergia_shared
        LIBRARY DESTINATION ${INSTALL_PATH}/lib
        ARCHIVE DESTINATION ${INSTALL_PATH}//lib
        )
