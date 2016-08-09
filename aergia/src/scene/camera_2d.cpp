#include <scene/camera_2d.h>
#include <utils/open_gl.h>

#include <ponos.h>

namespace aergia {

  Camera2D::Camera2D() {
    pos = vec2(0.f, 0.f);
    zoom = 1.f;
  }

  void Camera2D::look() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float pm[16];
    projection.matrix().row_major(pm);
    glMultMatrixf(pm);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //glMultMatrixf(view.c_matrix());
    //glMultMatrixf(model.c_matrix());
  }

  void Camera2D::resize(float w, float h) {
    display = vec2(w, h);
    ratio = w / h;
    clipSize = vec2(zoom, zoom);
    if(w < h)
    clipSize.y = clipSize.x / ratio;
    else clipSize.x = clipSize.y * ratio;
    update();
  }

  void Camera2D::setZoom(float z) {
    zoom = z;
    update();
  }

  void Camera2D::setPos(vec2 p) {
    pos = p;
    update();
  }

  void Camera2D::update() {
		projection = ponos::ortho(pos.x - clipSize.x, pos.x + clipSize.x,
							    pos.y - clipSize.y, pos.y + clipSize.y, -1.f, 1.f);
	}

  Transform Camera2D::getTransform() {
    model.computeInverse();
    view.computeInverse();
    projection.computeInverse();
    return model * view * projection;
  }

} // aergia namespace
