#include <aergia/scene/camera_2d.h>
#include <aergia/utils/open_gl.h>

#include <ponos/ponos.h>

using namespace ponos;

namespace aergia {

Camera2D::Camera2D() {
  this->zoom = 1.f;
  this->projection.reset(new OrthographicProjection());
}

/*void Camera2D::setZoom(float z) {
  zoom = z;
  resize(display.x, display.y);
}

}*/

void Camera2D::update() {
  auto clipSize = this->projection->clipSize;
  dynamic_cast<OrthographicProjection *>(this->projection.get())->set(
      pos.x - clipSize.x * zoom, pos.x + clipSize.x * zoom,
      pos.y - clipSize.y * zoom, pos.y + clipSize.y * zoom);
  model.computeInverse();
  view.computeInverse();
  projection->transform.computeInverse();
}

ponos::Ray3 Camera2D::pickRay(ponos::Point2 p) const {
  ponos::Point3 P = ponos::inverse(model * view)(ponos::inverse(projection->transform) *
      ponos::Point3(p.x, p.y, 1.f));
  // ponos::Point3 position(pos.x, pos.y, 0.f);
  return ponos::Ray3(P, vec3(0, 0, -1.f));
}

ponos::Line Camera2D::viewLineFromWindow(ponos::Point2 p) const {
  ponos::Point3 P = ponos::inverse(model * view)(ponos::inverse(projection->transform) *
      ponos::Point3(p.x, p.y, 1.f));
  return ponos::Line(P, vec3(0, 0, -1.f));
}

void Camera2D::fit(const ponos::BBox2D &b, float delta) {
  //setPos(ponos::vec2(b.center()));
  setZoom((b.size(b.maxExtent()) / 2.f) * delta);
  update();
}

ponos::Plane Camera2D::viewPlane(ponos::Point3 p) const {
  UNUSED_VARIABLE(p);
  return {ponos::Normal(0, 0, 1), 0};
}

} // aergia namespace
