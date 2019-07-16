#include <circe/colors/color.h>
#include <circe/colors/color_palette.h>
#include <circe/graphics/compute_shader.h>
#include <circe/graphics/shader.h>
#include <circe/graphics/shader_manager.h>
#include <circe/helpers/bvh_model.h>
#include <circe/helpers/camera_model.h>
#include <circe/helpers/cartesian_grid.h>
#include <circe/helpers/geometry_drawers.h>
#include <circe/helpers/grid_model.h>
#include <circe/helpers/hemesh_model.h>
#include <circe/helpers/quad_tree_model.h>
#include <circe/helpers/scene_handle.h>
#include <circe/helpers/tmesh_model.h>
#include <circe/helpers/vector_grid.h>
#include <circe/io/buffer.h>
#include <circe/io/display_renderer.h>
#include <circe/io/font_texture.h>
#include <circe/io/framebuffer.h>
#include <circe/io/graphics_display.h>
#include <circe/io/image_texture.h>
#include <circe/io/render_texture.h>
#include <circe/io/screen_quad.h>
#include <circe/io/storage_buffer.h>
#include <circe/io/texture.h>
#include <circe/io/texture_parameters.h>
#include <circe/io/viewport_display.h>
#include <circe/scene/bvh.h>
#include <circe/scene/camera.h>
#include <circe/scene/instance_set.h>
#include <circe/scene/mesh_utils.h>
#include <circe/scene/quad.h>
#include <circe/scene/scene.h>
#include <circe/scene/scene_mesh.h>
#include <circe/scene/scene_object.h>
#include <circe/scene/triangle_mesh.h>
#include <circe/scene/volume_box.h>
#include <circe/scene/wireframe_mesh.h>
#include <circe/ui/app.h>
#include <circe/ui/font_manager.h>
#include <circe/ui/interactive_object_interface.h>
#include <circe/ui/modifier_cursor.h>
#include <circe/ui/scene_app.h>
#include <circe/ui/text_object.h>
#include <circe/ui/text_renderer.h>
#include <circe/ui/track_mode.h>
#include <circe/ui/trackball.h>
#include <circe/ui/trackball_interface.h>
#include <circe/utils/open_gl.h>
#include <circe/utils/win32_utils.h>


namespace circe {

bool initialize();

} // namespace circe
