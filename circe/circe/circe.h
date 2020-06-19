#include <circe/colors/color.h>
#include <circe/colors/color_palette.h>
#include <circe/gl/graphics/compute_shader.h>
#include <circe/gl/graphics/shader.h>
#include <circe/gl/graphics/shader_manager.h>
#include <circe/gl/helpers/bvh_model.h>
#include <circe/gl/helpers/camera_model.h>
#include <circe/gl/helpers/cartesian_grid.h>
#include <circe/gl/helpers/geometry_drawers.h>
#include <circe/gl/helpers/grid_model.h>
//#include <circe/helpers/hemesh_model.h>
#include <circe/gl/helpers/quad_tree_model.h>
#include <circe/gl/helpers/scene_handle.h>
//#include <circe/helpers/tmesh_model.h>
#include <circe/gl/helpers/vector_grid.h>
#include <circe/gl/io/buffer.h>
#include <circe/gl/io/display_renderer.h>
#include <circe/gl/io/font_texture.h>
#include <circe/gl/io/framebuffer.h>
#include <circe/gl/io/graphics_display.h>
#include <circe/gl/io/image_texture.h>
#include <circe/gl/io/render_texture.h>
#include <circe/gl/io/screen_quad.h>
#include <circe/gl/io/storage_buffer.h>
#include <circe/gl/io/texture.h>
#include <circe/gl/io/texture_parameters.h>
#include <circe/gl/io/viewport_display.h>
#include <circe/gl/scene/bvh.h>
#include <circe/scene/camera_interface.h>
#include <circe/gl/scene/instance_set.h>
#include <circe/gl/scene/mesh_utils.h>
#include <circe/gl/scene/quad.h>
#include <circe/gl/scene/scene.h>
#include <circe/gl/scene/scene_mesh.h>
#include <circe/gl/scene/scene_object.h>
#include <circe/gl/scene/triangle_mesh.h>
#include <circe/gl/scene/volume_box.h>
#include <circe/gl/scene/wireframe_mesh.h>
#include <circe/gl/ui/app.h>
#include <circe/gl/ui/font_manager.h>
#include <circe/gl/ui/interactive_object_interface.h>
#include <circe/gl/ui/modifier_cursor.h>
#include <circe/gl/ui/scene_app.h>
#include <circe/gl/ui/text_object.h>
#include <circe/gl/ui/text_renderer.h>
#include <circe/ui/track_mode.h>
#include <circe/ui/trackball.h>
#include <circe/ui/trackball_interface.h>
#include <circe/gl/utils/open_gl.h>
#include <circe/gl/utils/win32_utils.h>
#include <circe/gl/utils/base_app.h>


namespace circe {

bool initialize();

} // namespace circe
