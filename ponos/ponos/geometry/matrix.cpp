#include <ponos/geometry/matrix.h>

#include <cmath>
#include <cstring>
#include <iostream>

namespace ponos {

Matrix4x4::Matrix4x4() {
  memset(m, 0, sizeof(m));
  for (int i = 0; i < 4; i++)
    m[i][i] = 1.f;
}

Matrix4x4::Matrix4x4(float mat[16], bool rowMajor) {
  if (rowMajor) {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        m[i][j] = mat[i * 4 + j];
  } else {
    for (int j = 0; j < 4; j++)
      for (int i = 0; i < 4; i++)
        m[j][i] = mat[i * 4 + j];
  }
}

Matrix4x4::Matrix4x4(float mat[4][4]) {
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      m[i][j] = mat[i][j];
}

Matrix4x4::Matrix4x4(float m00, float m01, float m02, float m03, float m10,
                     float m11, float m12, float m13, float m20, float m21,
                     float m22, float m23, float m30, float m31, float m32,
                     float m33) {
  m[0][0] = m00;
  m[0][1] = m01;
  m[0][2] = m02;
  m[0][3] = m03;
  m[1][0] = m10;
  m[1][1] = m11;
  m[1][2] = m12;
  m[1][3] = m13;
  m[2][0] = m20;
  m[2][1] = m21;
  m[2][2] = m22;
  m[2][3] = m23;
  m[3][0] = m30;
  m[3][1] = m31;
  m[3][2] = m32;
  m[3][3] = m33;
}

void Matrix4x4::setIdentity() {
  memset(m, 0, sizeof(m));
  for (int i = 0; i < 4; i++)
    m[i][i] = 1.f;
}

std::ostream &operator<<(std::ostream &os, const Matrix4x4 &m) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++)
      os << m.m[i][j] << " ";
    os << std::endl;
  }
  return os;
}

Matrix4x4 transpose(const Matrix4x4 &m) {
  return Matrix4x4(m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0], m.m[0][1],
                   m.m[1][1], m.m[2][1], m.m[3][1], m.m[0][2], m.m[1][2],
                   m.m[2][2], m.m[3][2], m.m[0][3], m.m[1][3], m.m[2][3],
                   m.m[3][3]);
}

// function extracted from MESA implementation of the GLU library
bool gluInvertMatrix(const float m[16], float invOut[16]) {
  float inv[16], det;
  int i;

  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
           m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];

  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
           m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];

  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
           m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];

  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
            m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
           m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];

  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
           m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];

  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
           m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];

  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
            m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
           m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];

  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
           m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];

  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
            m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];

  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
            m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
           m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];

  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
           m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];

  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
            m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];

  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
            m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

  det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

  if (det == 0)
    return false;

  det = 1.0 / det;

  for (i = 0; i < 16; i++)
    invOut[i] = inv[i] * det;

  return true;
}

Matrix4x4 inverse(const Matrix4x4 &m) {
  Matrix4x4 r;
  float mm[16], inv[16];
  m.row_major(mm);
  if (gluInvertMatrix(mm, inv)) {
    int k = 0;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        r.m[i][j] = inv[k++];
    return r;
  }

  float det = m.m[0][0] * m.m[1][1] * m.m[2][2] * m.m[3][3] +
              m.m[1][2] * m.m[2][3] * m.m[3][1] * m.m[1][3] +
              m.m[2][1] * m.m[3][2] * m.m[1][1] * m.m[2][3] +
              m.m[3][2] * m.m[1][2] * m.m[2][1] * m.m[3][3] +
              m.m[1][3] * m.m[2][2] * m.m[3][1] * m.m[0][1] +
              m.m[0][1] * m.m[2][3] * m.m[3][2] * m.m[0][2] +
              m.m[2][1] * m.m[3][3] * m.m[0][3] * m.m[2][2] +
              m.m[3][1] * m.m[0][1] * m.m[2][2] * m.m[3][3] +
              m.m[0][2] * m.m[2][3] * m.m[3][1] * m.m[0][3] +
              m.m[2][1] * m.m[3][2] * m.m[0][2] * m.m[0][1] +
              m.m[1][2] * m.m[3][3] * m.m[0][2] * m.m[1][3] +
              m.m[3][1] * m.m[0][3] * m.m[1][1] * m.m[3][2] -
              m.m[0][1] * m.m[1][3] * m.m[3][2] * m.m[0][2] -
              m.m[1][1] * m.m[3][3] * m.m[0][3] * m.m[1][2] -
              m.m[3][1] * m.m[0][3] * m.m[0][1] * m.m[1][3] -
              m.m[2][2] * m.m[0][2] * m.m[1][1] * m.m[2][3] -
              m.m[0][3] * m.m[1][2] * m.m[2][1] * m.m[0][1] -
              m.m[1][2] * m.m[2][3] * m.m[0][2] * m.m[1][3] -
              m.m[2][1] * m.m[0][3] * m.m[1][1] * m.m[2][2] -
              m.m[1][0] * m.m[1][0] * m.m[2][3] * m.m[3][2] -
              m.m[1][2] * m.m[2][0] * m.m[3][3] * m.m[1][3] -
              m.m[2][2] * m.m[3][0] * m.m[1][0] * m.m[2][2] -
              m.m[3][3] * m.m[1][2] * m.m[2][3] * m.m[3][0] -
              m.m[1][3] * m.m[2][0] * m.m[3][2] * m.m[1][1];
  if (fabs(det) < 1e-8)
    return r;

  r.m[0][0] =
      (m.m[1][1] * m.m[2][2] * m.m[3][3] + m.m[1][2] * m.m[2][3] * m.m[3][1] +
       m.m[1][3] * m.m[2][1] * m.m[3][2] - m.m[1][1] * m.m[2][3] * m.m[3][2] -
       m.m[1][2] * m.m[2][1] * m.m[3][3] - m.m[1][3] * m.m[2][2] * m.m[3][1]) /
      det;
  r.m[0][1] =
      (m.m[0][1] * m.m[2][3] * m.m[3][2] + m.m[0][2] * m.m[2][1] * m.m[3][3] +
       m.m[0][3] * m.m[2][2] * m.m[3][1] - m.m[0][1] * m.m[2][2] * m.m[3][3] -
       m.m[0][2] * m.m[2][3] * m.m[3][1] - m.m[0][3] * m.m[2][1] * m.m[3][2]) /
      det;
  r.m[0][2] =
      (m.m[0][1] * m.m[1][2] * m.m[3][3] + m.m[0][2] * m.m[1][3] * m.m[3][1] +
       m.m[0][3] * m.m[1][1] * m.m[3][2] - m.m[0][1] * m.m[1][3] * m.m[3][2] -
       m.m[0][2] * m.m[1][1] * m.m[3][3] - m.m[0][3] * m.m[1][2] * m.m[3][1]) /
      det;
  r.m[0][3] =
      (m.m[0][1] * m.m[1][3] * m.m[2][2] + m.m[0][2] * m.m[1][1] * m.m[2][3] +
       m.m[0][3] * m.m[1][2] * m.m[2][1] - m.m[0][1] * m.m[1][2] * m.m[2][3] -
       m.m[0][2] * m.m[1][3] * m.m[2][1] - m.m[0][3] * m.m[1][1] * m.m[2][2]) /
      det;
  r.m[1][0] =
      (m.m[1][0] * m.m[2][3] * m.m[3][2] + m.m[1][2] * m.m[2][0] * m.m[3][3] +
       m.m[1][3] * m.m[2][2] * m.m[3][0] - m.m[1][0] * m.m[2][2] * m.m[3][3] -
       m.m[1][2] * m.m[2][3] * m.m[3][0] - m.m[1][3] * m.m[2][0] * m.m[3][2]) /
      det;
  r.m[1][1] =
      (m.m[0][0] * m.m[2][2] * m.m[3][3] + m.m[0][2] * m.m[2][3] * m.m[3][0] +
       m.m[0][3] * m.m[2][0] * m.m[3][2] - m.m[0][0] * m.m[2][3] * m.m[3][2] -
       m.m[0][2] * m.m[2][0] * m.m[3][3] - m.m[0][3] * m.m[2][2] * m.m[3][0]) /
      det;
  r.m[1][2] =
      (m.m[0][0] * m.m[1][3] * m.m[3][2] + m.m[0][2] * m.m[1][0] * m.m[3][3] +
       m.m[0][3] * m.m[1][2] * m.m[3][0] - m.m[0][0] * m.m[1][2] * m.m[3][3] -
       m.m[0][2] * m.m[1][3] * m.m[3][0] - m.m[0][3] * m.m[1][0] * m.m[3][2]) /
      det;
  r.m[1][3] =
      (m.m[0][0] * m.m[1][2] * m.m[2][3] + m.m[0][2] * m.m[1][3] * m.m[2][0] +
       m.m[0][3] * m.m[1][0] * m.m[2][2] - m.m[0][0] * m.m[1][3] * m.m[2][2] -
       m.m[0][2] * m.m[1][0] * m.m[2][3] - m.m[0][3] * m.m[1][2] * m.m[2][0]) /
      det;
  r.m[2][0] =
      (m.m[1][0] * m.m[2][1] * m.m[3][3] + m.m[1][1] * m.m[2][3] * m.m[3][0] +
       m.m[1][3] * m.m[2][0] * m.m[3][1] - m.m[1][0] * m.m[2][3] * m.m[3][1] -
       m.m[1][1] * m.m[2][0] * m.m[3][3] - m.m[1][3] * m.m[2][1] * m.m[3][0]) /
      det;
  r.m[2][1] =
      (m.m[0][0] * m.m[2][3] * m.m[3][1] + m.m[0][1] * m.m[2][0] * m.m[3][3] +
       m.m[0][3] * m.m[2][1] * m.m[3][0] - m.m[0][0] * m.m[2][1] * m.m[3][3] -
       m.m[0][1] * m.m[2][3] * m.m[3][0] - m.m[0][3] * m.m[2][0] * m.m[3][1]) /
      det;
  r.m[2][2] =
      (m.m[0][0] * m.m[1][1] * m.m[3][3] + m.m[0][1] * m.m[1][3] * m.m[3][0] +
       m.m[0][3] * m.m[1][0] * m.m[3][1] - m.m[0][0] * m.m[1][3] * m.m[3][1] -
       m.m[0][1] * m.m[1][0] * m.m[3][3] - m.m[0][3] * m.m[1][1] * m.m[3][0]) /
      det;
  r.m[2][3] =
      (m.m[0][0] * m.m[1][3] * m.m[2][1] + m.m[0][1] * m.m[1][0] * m.m[2][3] +
       m.m[0][3] * m.m[1][1] * m.m[2][0] - m.m[0][0] * m.m[1][1] * m.m[2][3] -
       m.m[0][1] * m.m[1][3] * m.m[2][0] - m.m[0][3] * m.m[1][0] * m.m[2][1]) /
      det;
  r.m[3][0] =
      (m.m[1][0] * m.m[2][2] * m.m[3][1] + m.m[1][1] * m.m[2][0] * m.m[3][2] +
       m.m[1][2] * m.m[2][1] * m.m[3][0] - m.m[1][0] * m.m[2][1] * m.m[3][2] -
       m.m[1][1] * m.m[2][2] * m.m[3][0] - m.m[1][2] * m.m[2][0] * m.m[3][1]) /
      det;
  r.m[3][1] =
      (m.m[0][0] * m.m[2][1] * m.m[3][2] + m.m[0][1] * m.m[2][2] * m.m[3][0] +
       m.m[0][2] * m.m[2][0] * m.m[3][1] - m.m[0][0] * m.m[2][2] * m.m[3][1] -
       m.m[0][1] * m.m[2][0] * m.m[3][2] - m.m[0][2] * m.m[2][1] * m.m[3][0]) /
      det;
  r.m[3][2] =
      (m.m[0][0] * m.m[1][2] * m.m[3][1] + m.m[0][1] * m.m[1][0] * m.m[3][2] +
       m.m[0][2] * m.m[1][1] * m.m[3][0] - m.m[0][0] * m.m[1][1] * m.m[3][2] -
       m.m[0][1] * m.m[1][2] * m.m[3][0] - m.m[0][2] * m.m[1][0] * m.m[3][1]) /
      det;
  r.m[3][3] =
      (m.m[0][0] * m.m[1][1] * m.m[2][2] + m.m[0][1] * m.m[1][2] * m.m[2][0] +
       m.m[0][2] * m.m[1][0] * m.m[2][1] - m.m[0][0] * m.m[1][2] * m.m[2][1] -
       m.m[0][1] * m.m[1][0] * m.m[2][2] - m.m[0][2] * m.m[1][1] * m.m[2][0]) /
      det;

  return r;
}

void decompose(const Matrix4x4 &m, Matrix4x4 &r, Matrix4x4 &s) {
  // extract rotation r from transformation matrix
  float norm;
  int count = 0;
  r = m;
  do {
    // compute next matrix in series
    Matrix4x4 Rnext;
    Matrix4x4 Rit = inverse(transpose(r));
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        Rnext.m[i][j] = .5f * (r.m[i][j] + Rit.m[i][j]);
    // compute norm difference between R and Rnext
    norm = 0.f;
    for (int i = 0; i < 3; i++) {
      float n = fabsf(r.m[i][0] - Rnext.m[i][0]) +
                fabsf(r.m[i][1] - Rnext.m[i][1]) +
                fabsf(r.m[i][2] - Rnext.m[i][2]);
      norm = std::max(norm, n);
    }
  } while (++count < 100 && norm > .0001f);
  // compute scale S using rotation and original matrix
  s = Matrix4x4::mul(inverse(r), m);
}

Matrix3x3::Matrix3x3() {
  memset(m, 0, sizeof(m));
  for (int i = 0; i < 3; i++)
    m[i][i] = 1.f;
}

Matrix3x3::Matrix3x3(vec3 a, vec3 b, vec3 c)
    : Matrix3x3(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z) {}

Matrix3x3::Matrix3x3(float m00, float m01, float m02, float m10, float m11,
                     float m12, float m20, float m21, float m22) {
  m[0][0] = m00;
  m[0][1] = m01;
  m[0][2] = m02;
  m[1][0] = m10;
  m[1][1] = m11;
  m[1][2] = m12;
  m[2][0] = m20;
  m[2][1] = m21;
  m[2][2] = m22;
}

void Matrix3x3::setIdentity() {
  memset(m, 0, sizeof(m));
  for (int i = 0; i < 3; i++)
    m[i][i] = 1.f;
}

Matrix3x3 inverse(const Matrix3x3 &m) {
  Matrix3x3 r;
  float det =
      m.m[0][0] * m.m[1][1] * m.m[2][2] + m.m[1][0] * m.m[2][1] * m.m[0][2] +
      m.m[2][0] * m.m[0][1] * m.m[1][2] - m.m[0][0] * m.m[2][1] * m.m[1][2] -
      m.m[2][0] * m.m[1][1] * m.m[0][2] - m.m[1][0] * m.m[0][1] * m.m[2][2];
  if (fabs(det) < 1e-8)
    return r;

  r.m[0][0] = (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]) / det;
  r.m[0][1] = (m.m[0][2] * m.m[2][1] - m.m[0][1] * m.m[2][2]) / det;
  r.m[0][2] = (m.m[0][1] * m.m[1][2] - m.m[0][2] * m.m[1][1]) / det;
  r.m[1][0] = (m.m[1][2] * m.m[2][0] - m.m[1][0] * m.m[2][2]) / det;
  r.m[1][1] = (m.m[0][0] * m.m[2][2] - m.m[0][2] * m.m[2][0]) / det;
  r.m[1][2] = (m.m[0][2] * m.m[1][0] - m.m[0][0] * m.m[1][2]) / det;
  r.m[2][0] = (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]) / det;
  r.m[2][1] = (m.m[0][1] * m.m[2][0] - m.m[0][0] * m.m[2][1]) / det;
  r.m[2][2] = (m.m[0][0] * m.m[1][1] - m.m[0][1] * m.m[1][0]) / det;
  return r;
}

Matrix3x3 transpose(const Matrix3x3 &m) {
  return Matrix3x3(m.m[0][0], m.m[1][0], m.m[2][0], m.m[0][1], m.m[1][1],
                   m.m[2][1], m.m[0][2], m.m[1][2], m.m[2][2]);
}

Matrix3x3 star(const Vector3 a) {
  return Matrix3x3(0, -a[2], a[1], a[2], 0, -a[0], -a[1], a[0], 0);
}
std::ostream &operator<<(std::ostream &os, const Matrix3x3 &m) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++)
      os << m.m[i][j] << " ";
    os << std::endl;
  }
  return os;
}

Matrix2x2::Matrix2x2() {
  memset(m, 0, sizeof(m));
  for (int i = 0; i < 2; i++)
    m[i][i] = 1.f;
}

Matrix2x2::Matrix2x2(float m00, float m01, float m10, float m11) {
  m[0][0] = m00;
  m[0][1] = m01;
  m[1][0] = m10;
  m[1][1] = m11;
}

void Matrix2x2::setIdentity() {
  memset(m, 0, sizeof(m));
  for (int i = 0; i < 2; i++)
    m[i][i] = 1.f;
}

Matrix2x2 inverse(const Matrix2x2 &m) {
  Matrix2x2 r;
  float det = m.m[0][0] * m.m[1][1] - m.m[0][1] * m.m[1][0];
  if (det == 0.f)
    return r;
  float k = 1.f / det;
  r.m[0][0] = m.m[1][1] * k;
  r.m[0][1] = -m.m[0][1] * k;
  r.m[1][0] = -m.m[1][0] * k;
  r.m[1][1] = m.m[0][0] * k;
  return r;
}

Matrix2x2 transpose(const Matrix2x2 &m) {
  return Matrix2x2(m.m[0][0], m.m[1][0], m.m[0][1], m.m[1][1]);
}

std::ostream &operator<<(std::ostream &os, const Matrix2x2 &m) {
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++)
      os << m.m[i][j] << " ";
    os << std::endl;
  }
  return os;
}

} // ponos namespace
