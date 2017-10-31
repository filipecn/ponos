project(AERGIA)

set(HEADERS
	external/ply.h
	src/colors/color.h
	src/colors/color_palette.h
	src/graphics/shader.h
	src/graphics/shader_manager.h
	src/helpers/bvh_model.h
	src/helpers/camera_model.h
	src/helpers/cartesian_grid.h
	src/helpers/geometry_drawers.h
	src/helpers/grid_model.h
	src/helpers/hemesh_model.h
	src/helpers/quad_tree_model.h
	src/helpers/scene_handle.h
	src/helpers/vector_grid.h
	src/io/buffer.h
	src/io/framebuffer.h
	src/io/graphics_display.h
  src/io/procedural_texture.h
	src/io/texture.h
  src/io/texture_parameters.h
	src/io/utils.h
	src/io/viewport_display.h
	#  src/io/vulkan_application.h
	# src/io/vulkan_resource.h
	#  src/io/vulkan_utils.h
  src/scene/bvh.h
	src/scene/camera.h
	src/scene/camera_2d.h
	src/scene/camera_projection.h
	src/scene/mesh_utils.h
	src/scene/quad.h
	src/scene/scene.h
	src/scene/scene_object.h
	src/scene/triangle_mesh.h
	src/scene/wireframe_mesh.h
	src/ui/app.h
	src/ui/interactive_object_interface.h
	src/ui/modifier_cursor.h
	src/ui/scene_app.h
	src/ui/text.h
	src/ui/trackball.h
	src/ui/trackball_interface.h
	src/ui/track_mode.h
	src/utils/open_gl.h
	src/utils/win32_utils.h
	src/aergia.h
	)

set(SOURCES
	external/plyfile.c
	src/graphics/shader.cpp
	src/graphics/shader_manager.cpp
	src/helpers/cartesian_grid.cpp
	src/helpers/geometry_drawers.cpp
	src/io/framebuffer.cpp
	src/io/graphics_display.cpp
  src/io/procedural_texture.cpp
	src/io/texture.cpp
	src/io/utils.cpp
	src/io/viewport_display.cpp
	#  src/io/vulkan_application.cpp
  src/scene/bvh.cpp
	src/scene/camera.cpp
	src/scene/camera_2d.cpp
	src/scene/mesh_utils.cpp
	src/scene/quad.cpp
	src/scene/triangle_mesh.cpp
	src/scene/wireframe_mesh.cpp
	src/ui/app.cpp
	src/ui/text.cpp
	src/ui/trackball_interface.cpp
	src/utils/open_gl.cpp
	src/aergia.cpp
	)

include_directories("${AERGIA_SOURCE_DIR}/external")
include_directories(${AERGIA_INCLUDES})

if(WIN32)
	link_libraries(${GLFW_LIB} ${VULKAN_LIBRARY} opengl32 glu32)
else(WIN32)
	include_directories(${TINYOBJLOADER_INCLUDE_DIRS})
	link_libraries(${AERGIA_LIBRARIES} ${TINYOBJLOADER_LIBRARIES})
endif(WIN32)

link_libraries(ponos)
include_directories("${PONOS_SOURCE_DIR}/src")
include_directories("${AERGIA_SOURCE_DIR}/src")

add_library(aergia ${SOURCES} ${HEADERS})
set_target_properties(aergia PROPERTIES
	OUTPUT_NAME "aergia"
	FOLDER "AERGIA")

add_subdirectory(examples)
