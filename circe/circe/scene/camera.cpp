#include <circe/io/graphics_display.h>
#include <circe/scene/camera.h>
#include <circe/utils/open_gl.h>

#include <ponos/ponos.h>

using namespace ponos;

namespace circe {
/*
Camera::Camera(CameraProjection *_projection) {
  this->pos = Point3(20.f, 0.f, 0.f);
  this->target = Point3(0.f, 0.f, 0.f);
  this->up = vec3(0.f, 1.f, 0.f);
  this->zoom = 1.f;
  if (_projection)
    projection.reset(_projection);
  else {
    projection.reset(new PerspectiveProjection(45.f));
    projection->znear = 0.1f;
    projection->zfar = 1000.f;
  }
}

void Camera::setUp(const ponos::vec3 &u) {
  up = u;
  update();
}

void Camera::setFov(float f) {
  dynamic_cast<PerspectiveProjection *>(projection.get())->fov = f;
  projection->update();
  update();
}

void Camera::update() {
  view = ponos::lookAtRH(pos, target, up);
  view.computeInverse();
  model.computeInverse();
  frustum.set(model * view * projection->transform);
}

ponos::Line Camera::viewLineFromWindow(ponos::Point2 p) const {
  // TODO it would be more intuitive to be calculated by using the inverse of
MVP
  ponos::vec3 dir = normalize(target - pos);
  ponos::vec3 left = normalize(cross(normalize(up), dir));
  ponos::vec3 new_up = normalize(cross(dir, left));
  float ta = near * tanf(fov / 2.f);
  float tb = near * tanf((fov / ratio) / 2.f);
  ponos::Point3 P = pos + dir * near - left * p.x * ta + new_up * p.y * tb;
  ponos::Point3 P = ponos::inverse(model * view)(
      ponos::inverse(projection->transform) * ponos::Point3(p.x, p.y, -1.f));
  // std::cout << "view line from plane\n";
  // std::cout << P;
  // std::cout << pos;
  return Line(pos, P - pos);
}

ponos::Ray3 Camera::pickRay(ponos::Point2 p) const {
  ponos::Point3 P = ponos::inverse(model * view)(
      ponos::inverse(projection->transform) * ponos::Point3(p.x, p.y, -1.f));
  return ponos::Ray3(pos, P - pos);
}

ponos::Plane Camera::viewPlane(ponos::Point3 p) const {
  ponos::vec3 n = pos - p;
  if (fabs(n.length()) < 1e-8)
    n = ponos::vec3(0, 0, 0);
  else
    n = ponos::normalize(n);
  return Plane(ponos::Normal3(n), ponos::dot(n, ponos::vec3(p.x, p.y, p.z)));
}*/

} // circe namespace
