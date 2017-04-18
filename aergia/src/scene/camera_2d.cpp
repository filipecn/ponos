#include <scene/camera_2d.h>
#include <utils/open_gl.h>

#include <ponos.h>

using namespace ponos;

namespace aergia {

Camera2D::Camera2D() {
  pos = vec2(0.f, 0.f);
  zoom = 1.f;
}

void Camera2D::look() {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  float pm[16];
  projection.matrix().column_major(pm);
  glMultMatrixf(pm);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  // glMultMatrixf(view.c_matrix());
  // glMultMatrixf(model.c_matrix());
}

void Camera2D::resize(float w, float h) {
  display = vec2(w, h);
  ratio = w / h;
  clipSize = vec2(1.f);
  if (w < h)
    clipSize.y = clipSize.x / ratio;
  else
    clipSize.x = clipSize.y * ratio;
  update();
}

void Camera2D::setZoom(float z) {
  zoom = z;
  resize(display.x, display.y);
}

void Camera2D::setPos(vec2 p) {
  pos = p;
  update();
}

void Camera2D::update() {
  projection = ponos::ortho(
      pos.x - clipSize.x * zoom, pos.x + clipSize.x * zoom,
      pos.y - clipSize.y * zoom, pos.y + clipSize.y * zoom, -1.f, 1.f);
  model.computeInverse();
  view.computeInverse();
  projection.computeInverse();
}

Transform Camera2D::getTransform() const { return model * view * projection; }

ponos::Ray3 Camera2D::pickRay(ponos::Point2 p) const {
  ponos::Point3 P = ponos::inverse(model * view)(ponos::inverse(projection) *
                                                 ponos::Point3(p.x, p.y, -1.f));
  ponos::Point3 position(pos.x, pos.y, 0.f);
  return ponos::Ray3(position, P - position);
}

void Camera2D::fit(const ponos::BBox2D &b, float delta) {
  setPos(ponos::vec2(b.center()));
  setZoom((b.size(b.maxExtent()) / 2.f) * delta);
  update();
}

} // aergia namespace
