#include <aergia/io/graphics_display.h>
#include <aergia/scene/camera.h>
#include <aergia/utils/open_gl.h>

#include <ponos/ponos.h>

using namespace ponos;

namespace aergia {

Camera::Camera(CameraProjection *_projection) {
  pos = Point3(20.f, 0.f, 0.f);
  target = Point3(0.f, 0.f, 0.f);
  up = vec3(0.f, 1.f, 0.f);
  zoom = 1.f;
  if (_projection)
    projection.reset(_projection);
  else
    projection.reset(new PerspectiveProjection(45.f));
  projection->znear = 0.1f;
  projection->zfar = 1000.f;
}

void Camera::look() {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  float pm[16];
  projection->transform.matrix().column_major(pm);
  glMultMatrixf(pm);
  // gluPerspective(45.f, ratio, 0.1f, 1000.f);
  // GLdouble projMatrix[16];
  // glGetDoublev(GL_PROJECTION_MATRIX, projMatrix);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  float vm[16];
  view.matrix().column_major(vm);
  // gluLookAt(pos.x, pos.y, pos.z,
  // 		target.x, target.y, target.z,
  //		0, 1, 0);
  glMultMatrixf(vm);
  model.matrix().column_major(vm);
  glMultMatrixf(vm);
}

void Camera::resize(float w, float h) {
  display = vec2(w, h);
  projection->ratio = w / h;
  projection->clipSize = vec2(zoom, zoom);
  if (w < h)
    projection->clipSize.y = projection->clipSize.x / projection->ratio;
  else
    projection->clipSize.x = projection->clipSize.y * projection->ratio;
  projection->update();
  update();
}

void Camera::setZoom(float z) {
  zoom = z;
  update();
}

void Camera::setPosition(Point3 p) {
  pos = p;
  update();
}

void Camera::setTarget(Point3 p) {
  target = p;
  update();
}

void Camera::setUp(const ponos::vec3 &u) {
  up = u;
  update();
}

void Camera::setFov(float f) {
  static_cast<PerspectiveProjection *>(projection.get())->fov = f;
  projection->update();
  update();
}

void Camera::setFar(float f) {
  projection->zfar = f;
  projection->update();
  update();
}

void Camera::setNear(float n) {
  projection->znear = n;
  projection->update();
  update();
}

void Camera::update() {
  view = ponos::lookAtRH(pos, target, up);
  view.computeInverse();
  model.computeInverse();
  frustum.set(model * view * projection->transform);
}

Transform Camera::getTransform() const {
  return model * view * projection->transform;
}

ponos::Line Camera::viewLineFromWindow(ponos::Point2 p) const {
  // TODO it would be more intuitively calculated by using the inverse of MVP
  /*ponos::vec3 dir = normalize(target - pos);
  ponos::vec3 left = normalize(cross(normalize(up), dir));
  ponos::vec3 new_up = normalize(cross(dir, left));
  float ta = near * tanf(fov / 2.f);
  float tb = near * tanf((fov / ratio) / 2.f);
  ponos::Point3 P = pos + dir * near - left * p.x * ta + new_up * p.y * tb;*/
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
  return Plane(ponos::Normal(n), ponos::dot(n, ponos::vec3(p.x, p.y, p.z)));
}

} // aergia namespace
