#include <ponos/geometry/transform.h>
#include <ponos/geometry/utils.h>

namespace ponos {

Transform2::Transform2(const mat3 &mat, const mat3 inv_mat)
    : m(mat), m_inv(inv_mat) {}

Transform2::Transform2(const bbox2 &bbox) {
  m.m[0][0] = bbox.upper[0] - bbox.lower[0];
  m.m[1][1] = bbox.upper[1] - bbox.lower[1];
  m.m[0][2] = bbox.lower[0];
  m.m[1][2] = bbox.lower[1];
  m_inv = inverse(m);
}

void Transform2::reset() { m.setIdentity(); }

void Transform2::translate(const vec2 &d) {
  // TODO update inverse and make a better implementarion
  UNUSED_VARIABLE(d);
}

void Transform2::scale(real_t x, real_t y) {
  // TODO update inverse and make a better implementarion
  UNUSED_VARIABLE(x);
  UNUSED_VARIABLE(y);
}

void Transform2::rotate(real_t angle) {
  real_t sin_a = sinf(RADIANS(angle));
  real_t cos_a = cosf(RADIANS(angle));
  mat3 M(cos_a, -sin_a, 0.f, sin_a, cos_a, 0.f, 0.f, 0.f, 1.f);
  vec2 t = getTranslate();
  m.m[0][2] = m.m[1][2] = 0;
  m = m * M;
  m.m[0][2] = t.x;
  m.m[1][2] = t.y;
  // TODO update inverse and make a better implementarion
  m_inv = inverse(m);
}

Transform2 rotate(real_t angle) {
  real_t sin_a = sinf(RADIANS(angle));
  real_t cos_a = cosf(RADIANS(angle));
  mat3 m(cos_a, -sin_a, 0.f, sin_a, cos_a, 0.f, 0.f, 0.f, 1.f);
  return Transform2(m, transpose(m));
}

Transform2 translate(const vec2 &v) {
  mat3 m(1.f, 0.f, v.x, 0.f, 1.f, v.y, 0.f, 0.f, 1.f);
  mat3 m_inv(1.f, 0.f, -v.x, 0.f, 1.f, -v.y, 0.f, 0.f, 1.f);
  return Transform2(m, m_inv);
}

Transform2 inverse(const Transform2 &t) { return Transform2(t.m_inv, t.m); }

Transform::Transform(const mat4 &mat) : m(mat), m_inv(inverse(mat)) {}

Transform::Transform(const mat4 &mat, const mat4 &inv_mat)
    : m(mat), m_inv(inv_mat) {}

Transform::Transform(const real_t mat[4][4]) {
  m = mat4(mat[0][0], mat[0][1], mat[0][2], mat[0][3], mat[1][0], mat[1][1],
           mat[1][2], mat[1][3], mat[2][0], mat[2][1], mat[2][2], mat[2][3],
           mat[3][0], mat[3][1], mat[3][2], mat[3][3]);
  m_inv = inverse(m);
}

Transform::Transform(const bbox3 &bbox) {
  m.m[0][0] = bbox.upper[0] - bbox.lower[0];
  m.m[1][1] = bbox.upper[1] - bbox.lower[1];
  m.m[2][2] = bbox.upper[2] - bbox.lower[2];
  m.m[0][3] = bbox.lower[0];
  m.m[1][3] = bbox.lower[1];
  m.m[2][3] = bbox.lower[2];
  m_inv = inverse(m);
}

void Transform::reset() { m.setIdentity(); }

void Transform::translate(const vec3 &d) {
  // TODO update inverse and make a better implementarion
  UNUSED_VARIABLE(d);
}

void Transform::scale(real_t x, real_t y, real_t z) {
  // TODO update inverse and make a better implementarion
  UNUSED_VARIABLE(x);
  UNUSED_VARIABLE(y);
  UNUSED_VARIABLE(z);
}

bool Transform::swapsHandedness() const {
  real_t det = (m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1])) -
               (m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0])) +
               (m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]));
  return det < 0;
}

Transform2 scale(real_t x, real_t y) {
  mat3 m(x, 0, 0, 0, y, 0, 0, 0, 1);
  mat3 inv(1.f / x, 0, 0, 0, 1.f / y, 0, 0, 0, 1);
  return Transform2(m, inv);
}

Transform segmentToSegmentTransform(point3 a, point3 b, point3 c, point3 d) {
  // Consider two bases a b e f and c d g h
  // TODO implement
  return Transform();
}

Transform inverse(const Transform &t) { return Transform(t.m_inv, t.m); }

Transform translate(const vec3 &d) {
  mat4 m(1.f, 0.f, 0.f, d.x, 0.f, 1.f, 0.f, d.y, 0.f, 0.f, 1.f, d.z, 0.f, 0.f,
         0.f, 1.f);
  mat4 m_inv(1.f, 0.f, 0.f, -d.x, 0.f, 1.f, 0.f, -d.y, 0.f, 0.f, 1.f, -d.z, 0.f,
             0.f, 0.f, 1.f);
  return Transform(m, m_inv);
}

Transform scale(real_t x, real_t y, real_t z) {
  mat4 m(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);
  mat4 inv(1.f / x, 0, 0, 0, 0, 1.f / y, 0, 0, 0, 0, 1.f / z, 0, 0, 0, 0, 1);
  return Transform(m, inv);
}

Transform rotateX(real_t angle) {
  real_t sin_a = sinf(RADIANS(angle));
  real_t cos_a = cosf(RADIANS(angle));
  mat4 m(1.f, 0.f, 0.f, 0.f, 0.f, cos_a, -sin_a, 0.f, 0.f, sin_a, cos_a, 0.f,
         0.f, 0.f, 0.f, 1.f);
  return Transform(m, transpose(m));
}

Transform rotateY(real_t angle) {
  real_t sin_a = sinf(RADIANS(angle));
  real_t cos_a = cosf(RADIANS(angle));
  mat4 m(cos_a, 0.f, sin_a, 0.f, 0.f, 1.f, 0.f, 0.f, -sin_a, 0.f, cos_a, 0.f,
         0.f, 0.f, 0.f, 1.f);
  return Transform(m, transpose(m));
}

Transform rotateZ(real_t angle) {
  real_t sin_a = sinf(RADIANS(angle));
  real_t cos_a = cosf(RADIANS(angle));
  mat4 m(cos_a, -sin_a, 0.f, 0.f, sin_a, cos_a, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f,
         0.f, 0.f, 0.f, 1.f);
  return Transform(m, transpose(m));
}

Transform rotate(real_t angle, const vec3 &axis) {
  vec3 a = normalize(axis);
  real_t s = sinf(RADIANS(angle));
  real_t c = cosf(RADIANS(angle));
  real_t m[4][4];

  m[0][0] = a.x * a.x + (1.f - a.x * a.x) * c;
  m[0][1] = a.x * a.y * (1.f - c) - a.z * s;
  m[0][2] = a.x * a.z * (1.f - c) + a.y * s;
  m[0][3] = 0;

  m[1][0] = a.x * a.y * (1.f - c) + a.z * s;
  m[1][1] = a.y * a.y + (1.f - a.y * a.y) * c;
  m[1][2] = a.y * a.z * (1.f - c) - a.x * s;
  m[1][3] = 0;

  m[2][0] = a.x * a.z * (1.f - c) - a.y * s;
  m[2][1] = a.y * a.z * (1.f - c) + a.x * s;
  m[2][2] = a.z * a.z + (1.f - a.z * a.z) * c;
  m[2][3] = 0;

  m[3][0] = 0;
  m[3][1] = 0;
  m[3][2] = 0;
  m[3][3] = 1;

  mat4 mat(m);
  return Transform(mat, transpose(mat));
}

Transform frustumTransform(real_t left, real_t right, real_t bottom, real_t top,
                           real_t near, real_t far) {
  real_t tnear = 2.f * near;
  real_t lr = right - left;
  real_t bt = top - bottom;
  real_t nf = far - near;
  real_t m[4][4];
  m[0][0] = tnear / lr;
  m[0][1] = 0.f;
  m[0][2] = (right + left) / lr;
  m[0][3] = 0.f;

  m[1][0] = 0.f;
  m[1][1] = tnear / bt;
  m[1][2] = (top + bottom) / bt;
  m[1][3] = 0.f;

  m[2][0] = 0.f;
  m[2][1] = 0.f;
  m[2][2] = (-far - near) / nf;
  m[2][3] = (-tnear * far) / nf;

  m[3][0] = 0.f;
  m[3][1] = 0.f;
  m[3][2] = -1.f;
  m[3][3] = 0.f;

  mat4 projection(m);
  return Transform(projection, inverse(projection));
}

Transform perspective(real_t fov, real_t aspect, real_t zNear, real_t zFar) {
  real_t xmax = zNear * tanf(RADIANS(fov / 2.f));
  real_t ymax = xmax / aspect;
  return frustumTransform(-xmax, xmax, -ymax, ymax, zNear, zFar);
}

Transform perspective(real_t fov, real_t n, real_t f) {
  // perform projectiev divide
  mat4 persp = mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, f / (f - n), -f * n / (f - n),
                    0, 0, 1, 0);
  // scale to canonical viewing volume
  real_t invTanAng = 1.f / tanf(RADIANS(fov) / 2.f);
  return scale(invTanAng, invTanAng, 1) * Transform(persp);
}

Transform lookAt(const point3 &pos, const point3 &target, const vec3 &up) {
  vec3 dir = normalize(target - pos);
  vec3 left = normalize(cross(normalize(up), dir));
  vec3 new_up = cross(dir, left);
  real_t m[4][4];
  m[0][0] = left.x;
  m[1][0] = left.y;
  m[2][0] = left.z;
  m[3][0] = 0;

  m[0][1] = new_up.x;
  m[1][1] = new_up.y;
  m[2][1] = new_up.z;
  m[3][1] = 0;

  m[0][2] = dir.x;
  m[1][2] = dir.y;
  m[2][2] = dir.z;
  m[3][2] = 0;

  m[0][3] = pos.x;
  m[1][3] = pos.y;
  m[2][3] = pos.z;
  m[3][3] = 1;

  mat4 cam_to_world(m);
  return Transform(inverse(cam_to_world), cam_to_world);
}

Transform lookAtRH(const point3 &pos, const point3 &target, const vec3 &up) {
  vec3 dir = normalize(pos - target);
  vec3 left = normalize(cross(normalize(up), dir));
  vec3 new_up = cross(dir, left);
  real_t m[4][4];
  m[0][0] = left.x;
  m[0][1] = left.y;
  m[0][2] = left.z;
  m[0][3] = -dot(left, vec3(pos - point3()));

  m[1][0] = new_up.x;
  m[1][1] = new_up.y;
  m[1][2] = new_up.z;
  m[1][3] = -dot(new_up, vec3(pos - point3()));

  m[2][0] = dir.x;
  m[2][1] = dir.y;
  m[2][2] = dir.z;
  m[2][3] = -dot(dir, vec3(pos - point3()));

  m[3][0] = 0;
  m[3][1] = 0;
  m[3][2] = 0;
  m[3][3] = 1;

  /*
    dir = normalize(target - pos);
    left = normalize(cross(normalize(up), dir));
    new_up = cross(dir, left);

    m[0][3] = pos.x;
    m[1][3] = pos.y;
    m[2][3] = pos.z;
    m[3][3] = 1;
    m[0][0] = left.x;
    m[1][0] = left.y;
    m[2][0] = left.z;
    m[3][0] = 0;
    m[0][1] = new_up.x;
    m[1][1] = new_up.y;
    m[2][1] = new_up.z;
    m[3][1] = 0;
    m[0][2] = dir.x;
    m[1][2] = dir.y;
    m[2][2] = dir.z;
    m[3][2] = 0;
  */
  mat4 cam_to_world(m);
  return Transform(cam_to_world, inverse(cam_to_world));
}

Transform ortho(real_t left, real_t right, real_t bottom, real_t top,
                real_t near, real_t far) {
  real_t m[4][4];

  m[0][0] = 2.f / (right - left);
  m[1][0] = 0.f;
  m[2][0] = 0.f;
  m[3][0] = 0.f;

  m[0][1] = 0.f;
  m[1][1] = 2.f / (top - bottom);
  m[2][1] = 0.f;
  m[3][1] = 0.f;

  m[0][2] = 0.f;
  m[1][2] = 0.f;
  m[2][2] = 2.f / (far - near);
  m[3][2] = 0.f;

  m[0][3] = -(right + left) / (right - left);
  m[1][3] = -(top + bottom) / (top - bottom);
  m[2][3] = -(far + near) / (far - near);
  m[3][3] = 1.f;

  mat4 projection(m);
  return Transform(projection, inverse(projection));
}

Transform orthographic(real_t znear, real_t zfar) {
  return scale(1.f, 1.f, 1.f / (zfar - znear)) *
         translate(vec3(0.f, 0.f, -znear));
}

} // namespace ponos
