project(AERGIA)

set(HEADERS
        external/ply.h
        aergia/colors/color.h
        aergia/colors/color_palette.h
        aergia/graphics/compute_shader.h
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
        aergia/io/buffer.h
        aergia/io/framebuffer.h
        aergia/io/graphics_display.h
        aergia/io/procedural_texture.h
        aergia/io/texture.h
        aergia/io/texture_parameters.h
        aergia/io/utils.h
        aergia/io/viewport_display.h
        #  aergia/io/vulkan_application.h
        # aergia/io/vulkan_resource.h
        #  aergia/io/vulkan_utils.h
        aergia/scene/bvh.h
        aergia/scene/camera.h
        aergia/scene/camera_2d.h
        aergia/scene/camera_projection.h
        aergia/scene/mesh_utils.h
        aergia/scene/quad.h
        aergia/scene/scene.h
        aergia/scene/scene_object.h
        aergia/scene/triangle_mesh.h
        aergia/scene/wireframe_mesh.h
        aergia/ui/app.h
        aergia/ui/interactive_object_interface.h
        aergia/ui/modifier_cursor.h
        aergia/ui/scene_app.h
        aergia/ui/text.h
        aergia/ui/trackball.h
        aergia/ui/trackball_interface.h
        aergia/ui/track_mode.h
        aergia/utils/open_gl.h
        aergia/utils/win32_utils.h
        aergia/aergia.h
        )

set(SOURCES
        external/plyfile.c
        aergia/graphics/compute_shader.cpp
        aergia/graphics/shader.cpp
        aergia/graphics/shader_manager.cpp
        aergia/helpers/cartesian_grid.cpp
        aergia/helpers/geometry_drawers.cpp
        aergia/io/framebuffer.cpp
        aergia/io/graphics_display.cpp
        aergia/io/procedural_texture.cpp
        aergia/io/texture.cpp
        aergia/io/utils.cpp
        aergia/io/viewport_display.cpp
        #  aergia/io/vulkan_application.cpp
        aergia/scene/bvh.cpp
        aergia/scene/camera.cpp
        aergia/scene/camera_2d.cpp
        aergia/scene/mesh_utils.cpp
        aergia/scene/quad.cpp
        aergia/scene/triangle_mesh.cpp
        aergia/scene/wireframe_mesh.cpp
        aergia/ui/app.cpp
        aergia/ui/text.cpp
        aergia/ui/trackball_interface.cpp
        aergia/utils/open_gl.cpp
        aergia/aergia.cpp
        )

include_directories("${AERGIA_SOURCE_DIR}/external")
include_directories(${AERGIA_INCLUDES})

if (WIN32)
    link_libraries(${GLFW_LIB} ${VULKAN_LIBRARY} opengl32 glu32)
else (WIN32)
    include_directories(${TINYOBJLOADER_INCLUDE_DIRS})
    link_libraries(${AERGIA_LIBRARIES} ${TINYOBJLOADER_LIBRARIES})
endif (WIN32)

link_libraries(ponos)
include_directories(${PONOS_SOURCE_DIR})
include_directories(${AERGIA_SOURCE_DIR})

if (NOT ("${FREETYPE_INCLUDE}" STREQUAL ""))
    add_definitions(-DFREETYPE_INCLUDED)
endif (NOT ("${FREETYPE_INCLUDE}" STREQUAL ""))

add_library(aergia ${SOURCES} ${HEADERS})
if (NOT WIN32)
    add_library(aergia_shared SHARED ${SOURCES} ${HEADERS})
endif (NOT WIN32)
set_target_properties(aergia PROPERTIES
        OUTPUT_NAME "aergia"
        FOLDER "AERGIA")

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
