#include <aergia/colors/color.h>
#include <aergia/colors/color_palette.h>
#include <aergia/graphics/compute_shader.h>
#include <aergia/graphics/shader.h>
#include <aergia/graphics/shader_manager.h>
#include <aergia/helpers/bvh_model.h>
#include <aergia/helpers/camera_model.h>
#include <aergia/helpers/cartesian_grid.h>
#include <aergia/helpers/geometry_drawers.h>
#include <aergia/helpers/grid_model.h>
#include <aergia/helpers/hemesh_model.h>
#include <aergia/helpers/quad_tree_model.h>
#include <aergia/helpers/scene_handle.h>
#include <aergia/helpers/tmesh_model.h>
#include <aergia/helpers/vector_grid.h>
#include <aergia/io/buffer.h>
#include <aergia/io/display_renderer.h>
#include <aergia/io/storage_buffer.h>
#include <aergia/io/framebuffer.h>
#include <aergia/io/graphics_display.h>
#include <aergia/io/image_texture.h>
#include <aergia/io/render_texture.h>
#include <aergia/io/texture.h>
#include <aergia/io/texture_parameters.h>
#include <aergia/io/viewport_display.h>
//#includ<aergia/ >io/vulkan_application.h>
//#includ<aergia/ >io/vulkan_resource.h>
//#includ<aergia/ >io/vulkan_utils.h>
#include <aergia/scene/bvh.h>
#include <aergia/scene/camera.h>
#include <aergia/scene/instance_set.h>
#include <aergia/scene/mesh_utils.h>
#include <aergia/scene/quad.h>
#include <aergia/scene/scene.h>
#include <aergia/scene/scene_mesh.h>
#include <aergia/scene/scene_object.h>
#include <aergia/scene/triangle_mesh.h>
#include <aergia/scene/wireframe_mesh.h>
#include <aergia/ui/app.h>
#include <aergia/ui/interactive_object_interface.h>
#include <aergia/ui/modifier_cursor.h>
#include <aergia/ui/scene_app.h>
#include <aergia/ui/text.h>
#include <aergia/ui/track_mode.h>
#include <aergia/ui/trackball.h>
#include <aergia/ui/trackball_interface.h>
#include <aergia/utils/open_gl.h>
#include <aergia/utils/win32_utils.h>

namespace aergia {

bool initialize();

} // aergia namespace
