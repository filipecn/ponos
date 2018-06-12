project(AERGIA)

set(HEADERS
        aergia/colors/color.h
        aergia/colors/color_palette.h
        aergia/graphics/compute_shader.h
        aergia/graphics/post_effect.h
        aergia/graphics/shader.h
        aergia/graphics/shader_manager.h
        aergia/helpers/bvh_model.h
        aergia/helpers/camera_model.h
        aergia/helpers/cartesian_grid.h
        aergia/helpers/geometry_drawers.h
        aergia/helpers/grid_model.h
        aergia/helpers/hemesh_model.h
        aergia/helpers/quad_tree_model.h
        aergia/helpers/scene_handle.h
        aergia/helpers/vector_grid.h
        aergia/helpers/tmesh_model.h
        aergia/io/buffer.h
        aergia/io/display_renderer.h
        aergia/io/framebuffer.h
        aergia/io/graphics_display.h
        aergia/io/image_texture.h
        aergia/io/font_texture.h
        aergia/io/render_texture.h
        aergia/io/storage_buffer.h
        aergia/io/texture.h
        aergia/io/texture_parameters.h
        aergia/io/utils.h
        aergia/io/viewport_display.h
        aergia/io/user_input.h
        #  aergia/io/vulkan_application.h
        # aergia/io/vulkan_resource.h
        #  aergia/io/vulkan_utils.h
        aergia/scene/bvh.h
        aergia/scene/camera.h
        # aergia/scene/camera_2d.h
        aergia/scene/camera_projection.h
        aergia/scene/mesh_utils.h
        aergia/scene/quad.h
        aergia/scene/scene.h
        aergia/scene/scene_mesh.h
        aergia/scene/scene_object.h
        aergia/scene/triangle_mesh.h
        aergia/scene/wireframe_mesh.h
        aergia/ui/app.h
        aergia/ui/interactive_object_interface.h
        aergia/ui/modifier_cursor.h
        aergia/ui/scene_app.h
        aergia/ui/text_renderer.h
        aergia/ui/font_manager.h
        aergia/ui/trackball.h
        aergia/ui/trackball_interface.h
        aergia/ui/track_mode.h
        aergia/ui/ui_camera.h
        aergia/utils/open_gl.h
        aergia/utils/win32_utils.h
        aergia/aergia.h
        )

set(SOURCES
        aergia/graphics/compute_shader.cpp
        aergia/graphics/post_effect.cpp
        aergia/graphics/shader.cpp
        aergia/graphics/shader_manager.cpp
        aergia/helpers/cartesian_grid.cpp
        aergia/helpers/geometry_drawers.cpp
        aergia/io/buffer.cpp
        aergia/io/display_renderer.cpp
        aergia/io/framebuffer.cpp
        aergia/io/graphics_display.cpp
        aergia/io/image_texture.cpp
        aergia/io/render_texture.cpp
        aergia/io/font_texture.cpp
        aergia/io/storage_buffer.cpp
        aergia/io/texture.cpp
        aergia/io/utils.cpp
        aergia/io/viewport_display.cpp
        #  aergia/io/vulkan_application.cpp
        aergia/scene/bvh.cpp
        aergia/scene/camera.cpp
        # aergia/scene/camera_2d.cpp
        aergia/scene/mesh_utils.cpp
        aergia/scene/quad.cpp
        aergia/scene/scene_mesh.cpp
        aergia/scene/triangle_mesh.cpp
        aergia/scene/wireframe_mesh.cpp
        aergia/ui/app.cpp
        aergia/ui/text_renderer.cpp
        aergia/ui/font_manager.cpp
        aergia/ui/trackball_interface.cpp
        aergia/ui/ui_camera.cpp
        aergia/utils/open_gl.cpp
        aergia/aergia.cpp
        aergia/scene/instance_set.cpp aergia/scene/instance_set.h)

if (WIN32)
    link_libraries(${GLFW_LIB} ${VULKAN_LIBRARY} opengl32 glu32)
endif (WIN32)

add_library(aergia ${SOURCES} ${HEADERS})
set_target_properties(aergia PROPERTIES
        OUTPUT_NAME "aergia"
        FOLDER "AERGIA")
target_compile_definitions(aergia PUBLIC -DASSET_PATH="${AERGIA_SOURCE_DIR}/examples/assets")
target_include_directories(aergia PUBLIC ${AERGIA_SOURCE_DIR} ${PLY_INCLUDES} ${STB_INCLUDES})
target_link_libraries(aergia PUBLIC ponos ${NANOGUI_LIBS} ${PLY_LIBS} ${TINYOBJLOADER_LIBS} GLEW GLU)

add_subdirectory(examples)

if (WIN32)
    SET(INSTALL_PATH ${ROOT_PROJECT_PATH}/wbuild)
else (WIN32)
    SET(INSTALL_PATH ${ROOT_PROJECT_PATH}/build)
endif (WIN32)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/aergia
        DESTINATION ${INSTALL_PATH}/include
        FILES_MATCHING REGEX "(.*\\.[inl|h])")

install(TARGETS aergia #aergia_shared
        LIBRARY DESTINATION ${INSTALL_PATH}/lib
        ARCHIVE DESTINATION ${INSTALL_PATH}//lib
        )
