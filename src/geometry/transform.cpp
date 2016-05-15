#include "geometry/transform.h"

#include "geometry/utils.h"

namespace ponos {

  Transform::Transform(const Matrix4x4& mat)
    : m(mat), m_inv(inverse(mat)) {}

  Transform::Transform(const Matrix4x4& mat, const Matrix4x4 inv_mat)
    : m(mat), m_inv(inv_mat) {}

  Transform::Transform(const float mat[4][4]) {
    m = Matrix4x4(mat[0][0], mat[0][1], mat[0][2], mat[0][3],
                  mat[1][0], mat[1][1], mat[1][2], mat[1][3],
                  mat[2][0], mat[2][1], mat[2][2], mat[2][3],
                  mat[3][0], mat[3][1], mat[3][2], mat[3][3]);
    m_inv = inverse(m);
  }

  bool Transform::swapsHandedness() const {
    float det = (m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1])) -
                (m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0])) +
                (m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]));
    return det < 0.f;
  }

  Transform inverse(const Transform& t) {
    return Transform(t.m_inv, t.m);
  }

  Transform translate(const Vector3 &d) {
    Matrix4x4 m(1.f, 0.f, 0.f, d.x,
                0.f, 1.f, 0.f, d.y,
                0.f, 0.f, 1.f, d.z,
                0.f, 0.f, 0.f, 1.f);
    Matrix4x4 m_inv(1.f, 0.f, 0.f, -d.x,
                    0.f, 1.f, 0.f, -d.y,
                    0.f, 0.f, 1.f, -d.z,
                    0.f, 0.f, 0.f, 1.f);
    return Transform(m, m_inv);
  }

  Transform rotateX(float angle) {
    float sin_a = sinf(TO_RADIANS(angle));
    float cos_a = cosf(TO_RADIANS(angle));
    Matrix4x4 m(1.f, 0.f,   0.f,    0.f,
                0.f, cos_a, -sin_a, 0.f,
                0.f, sin_a, cos_a,  0.f,
                0.f, 0.f,   0.f,    1.f);
    return Transform(m, transpose(m));
  }

  Transform rotateY(float angle) {
    float sin_a = sinf(TO_RADIANS(angle));
    float cos_a = cosf(TO_RADIANS(angle));
    Matrix4x4 m(cos_a,  0.f, sin_a, 0.f,
                0.f,    1.f, 0.f,   0.f,
                -sin_a, 0.f, cos_a, 0.f,
                0.f,    0.f, 0.f,   1.f);
    return Transform(m, transpose(m));
  }

  Transform rotateZ(float angle) {
    float sin_a = sinf(TO_RADIANS(angle));
    float cos_a = cosf(TO_RADIANS(angle));
    Matrix4x4 m(cos_a, -sin_a, 0.f, 0.f,
                sin_a, cos_a,  0.f, 0.f,
                0.f,   0.f,    1.f, 0.f,
                0.f,   0.f,    0.f, 1.f);
    return Transform(m, transpose(m));
  }

  Transform rotate(float angle, const Vector3& axis) {
    Vector3 a = normalize(axis);
    float s = sinf(TO_RADIANS(angle));
    float c = cosf(TO_RADIANS(angle));
    float m[4][4];

    m[0][0] = a.x * a.x + (1.f - a.x * a.x) * c;
    m[0][1] = a.x * a.y * (1.f - c) - a.z * s;
    m[0][2] = a.x * a.z * (1.f - c) + a.z * s;
    m[0][3] = 0;

    m[1][0] = a.x * a.y * (1.f - c) + a.z * s;
    m[1][1] = a.y * a.y + (1.f - a.y * a.y) * c;
    m[1][2] = a.y * a.z * (1.f - c) - a.x * s;
    m[1][3] = 0;

    m[2][0] = a.x * a.z * (1.f - c) - a.y * s;
    m[2][1] = a.y * a.z * (1.f - c) + a.x * s;
    m[2][2] = a.z * a.z + (1.f - a.z * a.z) * c;
    m[2][0] = 0;

    m[3][0] = 0;
    m[3][1] = 0;
    m[3][2] = 0;
    m[3][3] = 0;

    Matrix4x4 mat(m);
    return Transform(mat, transpose(mat));
  }

  Transform lookAt(const Point3& pos, const Point3& target, const Vector3& up) {
    Vector3 dir = normalize(target - pos);
    Vector3 left = normalize(cross(normalize(up), dir));
    Vector3 new_up = cross(dir, left);
    float m[4][4];
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

    Matrix4x4 cam_to_world(m);
    return Transform(inverse(cam_to_world), cam_to_world);
  }

}; // ponos namespace
