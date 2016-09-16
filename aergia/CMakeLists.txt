project(AERGIA)

set(CMAKE_MODULE_PATH "${AERGIA_SOURCE_DIR}/CMake/modules")

set(HEADERS
	src/graphics/shader.h
	src/graphics/shader_manager.h
	src/helpers/camera_model.h
	src/helpers/cartesian_grid.h
	src/helpers/geometry_drawers.h
	src/io/graphics_display.h
	src/io/utils.h
	src/io/viewport_display.h
	src/scene/camera.h
	src/scene/camera_2d.h
	src/scene/raw_mesh.h
	src/scene/scene.h
	src/scene/scene_object.h
	src/scene/triangle_mesh.h
	src/ui/app.h
	src/ui/scene_app.h
	src/ui/trackball.h
	src/ui/trackball_interface.h
	src/ui/track_mode.h
	src/utils/open_gl.h
	src/utils/win32_utils.h
	src/aergia.h
	)

set(SOURCES
	src/graphics/shader.cpp
	src/graphics/shader_manager.cpp
	src/helpers/cartesian_grid.cpp
	src/helpers/geometry_drawers.cpp
	src/io/graphics_display.cpp
	src/io/utils.cpp
	src/io/viewport_display.cpp
	src/scene/camera.cpp
	src/scene/camera_2d.cpp
	src/scene/triangle_mesh.cpp
	src/ui/app.cpp
	src/ui/trackball_interface.cpp
	src/utils/open_gl.cpp
	)

if(WIN32)
	find_library(GLFW_LIB NAMES glfw3 PATHS ${PROJECT_SOURCE_DIR}/external/GLFW)
	include_directories("${PROJECT_SOURCE_DIR}/external")
	find_library(VULKAN_LIB NAMES vulkan-1 vulkan)
	find_package(Vulkan REQUIRED)
	link_libraries(${GLFW_LIB} ${VULKAN_LIBRARY} opengl32 glu32)
else(WIN32)
	find_package(glfw3 REQUIRED)
	include_directories(${GLFW_INCLUDE_DIRS})
	find_package(OpenGL REQUIRED)
	include_directories(${OPENGL_INCLUDE_DIRS})
	link_libraries(glfw ${OPENGL_LIBRARIES})
	include_directories("/usr/local/include")
	link_libraries(tinyobjloader "/usr/local/lib/libtinyobjloader.a")
endif(WIN32)

link_libraries(ponos)
include_directories("${PONOS_SOURCE_DIR}/src")
include_directories("${AERGIA_SOURCE_DIR}/src")

add_library(aergia ${SOURCES} ${HEADERS})
set_target_properties(aergia PROPERTIES
	OUTPUT_NAME "aergia"
	FOLDER "AERGIA")

target_link_libraries(aergia ${aergia_LIBRARIES} GLEW)

add_subdirectory(examples)
