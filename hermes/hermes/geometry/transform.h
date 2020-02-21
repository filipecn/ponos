/*
 * Copyright (c) 2019 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#ifndef HERMES_GEOMETRY_CUDA_TRANSFORM_H
#define HERMES_GEOMETRY_CUDA_TRANSFORM_H

#include <cmath>
#include <hermes/geometry/bbox.h>
#include <hermes/geometry/matrix.h>
#include <hermes/geometry/vector.h>

namespace hermes {

namespace cuda {

template <typename T> class Transform2 {
public:
  __host__ __device__ Transform2() {
    m.setIdentity();
    m_inv.setIdentity();
  }
  __host__ __device__ Transform2(const Matrix3x3<T> &mat,
                                 const Matrix3x3<T> inv_mat)
      : m(mat), m_inv(inv_mat) {}
  __host__ __device__ Transform2(const BBox2<T> &bbox) {
    m.m[0][0] = bbox.upper[0] - bbox.lower[0];
    m.m[1][1] = bbox.upper[1] - bbox.lower[1];
    m.m[0][2] = bbox.lower[0];
    m.m[1][2] = bbox.lower[1];
    m_inv = inverse(m);
  }
  __host__ __device__ void reset() { m.setIdentity(); }
  __host__ __device__ void translate(const Vector2<T> &d) {
    // TODO update inverse and make a better implementarion
    UNUSED_VARIABLE(d);
  }
  __host__ __device__ void scale(T x, T y) {
    // TODO update inverse and make a better implementarion
    UNUSED_VARIABLE(x);
    UNUSED_VARIABLE(y);
  }
  __host__ __device__ void rotate(T angle) {
    T sin_a = sinf(TO_RADIANS(angle));
    T cos_a = cosf(TO_RADIANS(angle));
    Matrix3x3<T> M(cos_a, -sin_a, 0.f, sin_a, cos_a, 0.f, 0.f, 0.f, 1.f);
    Vector2<T> t = getTranslate();
    m.m[0][2] = m.m[1][2] = 0;
    m = m * M;
    m.m[0][2] = t.x;
    m.m[1][2] = t.y;
    // TODO update inverse and make a better implementation
    m_inv = inverse(m);
  }
  template <typename S>
  friend __host__ __device__ Transform2<S> inverse(const Transform2<S> &t);
  __host__ __device__ void operator()(const Point2<T> &p, Point2<T> *r) const {
    T x = p.x, y = p.y;
    r->x = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2];
    r->y = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2];
    T wp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2];
    if (wp != 1.f)
      *r /= wp;
  }
  __host__ __device__ void operator()(const Vector2<T> &v,
                                      Vector2<T> *r) const {
    T x = v.x, y = v.y;
    r->x = m.m[0][0] * x + m.m[0][1] * y;
    r->y = m.m[1][0] * x + m.m[1][1] * y;
  }
  __host__ __device__ Vector2<T> operator()(const Vector2<T> &v) const {
    T x = v.x, y = v.y;
    return Vector2<T>(m.m[0][0] * x + m.m[0][1] * y,
                      m.m[1][0] * x + m.m[1][1] * y);
  }
  __host__ __device__ Point2<T> operator()(const Point2<T> &p) const {
    T x = p.x, y = p.y;
    T xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2];
    T yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2];
    T wp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2];
    if (wp == 1.f)
      return Point2<T>(xp, yp);
    return Point2<T>(xp / wp, yp / wp);
  }
  __host__ __device__ BBox2<T> operator()(const BBox2<T> &b) const {
    const Transform2 &M = *this;
    BBox2<T> ret;
    ret = make_union(ret, M(Point2<T>(b.lower.x, b.lower.y)));
    ret = make_union(ret, M(Point2<T>(b.upper.x, b.lower.y)));
    ret = make_union(ret, M(Point2<T>(b.upper.x, b.upper.y)));
    ret = make_union(ret, M(Point2<T>(b.lower.x, b.upper.y)));
    return ret;
  }
  __host__ __device__ Transform2 operator*(const Transform2 &t) const {
    Matrix3x3<T> m1 = Matrix3x3<T>::mul(m, t.m);
    Matrix3x3<T> m1_inv = Matrix3x3<T>::mul(t.m_inv, m_inv);
    return Transform2(m1, m1_inv);
  }
  __host__ __device__ Vector2<T> getTranslate() const {
    return Vector2<T>(m.m[0][2], m.m[1][2]);
  }
  __host__ __device__ Vector2<T> getScale() const { return s; }
  __host__ __device__ void computeInverse() { m_inv = inverse(m); }
  __host__ __device__ Matrix3x3<T> getMatrix() const { return m; }

private:
  Matrix3x3<T> m, m_inv;
  Vector2<T> s;
};

template <typename T> __host__ __device__ Transform2<T> rotate(T angle) {
  T sin_a = sinf(TO_RADIANS(angle));
  T cos_a = cosf(TO_RADIANS(angle));
  Matrix3x3<T> m(cos_a, -sin_a, 0.f, sin_a, cos_a, 0.f, 0.f, 0.f, 1.f);
  return Transform2<T>(m, transpose(m));
}
template <typename T>
__host__ __device__ Transform2<T> translate(const Vector2<T> &v) {
  Matrix3x3<T> m(1.f, 0.f, v.x, 0.f, 1.f, v.y, 0.f, 0.f, 1.f);
  Matrix3x3<T> m_inv(1.f, 0.f, -v.x, 0.f, 1.f, -v.y, 0.f, 0.f, 1.f);
  return Transform2<T>(m, m_inv);
}
template <typename T>
__host__ __device__ Transform2<T> inverse(const Transform2<T> &t) {
  return Transform2<T>(t.m_inv, t.m);
}
template <typename T> __host__ __device__ Transform2<T> scale(T x, T y) {
  Matrix3x3<T> m(x, 0, 0, 0, y, 0, 0, 0, 1);
  Matrix3x3<T> inv(1.f / x, 0, 0, 0, 1.f / y, 0, 0, 0, 1);
  return Transform2<T>(m, inv);
}
template <typename T>
__host__ __device__ Transform2<T> scale(const Vector2<T> &v) {
  Matrix3x3<T> m(v.x, 0, 0, 0, v.y, 0, 0, 0, 1);
  Matrix3x3<T> inv(1.f / v.x, 0, 0, 0, 1.f / v.y, 0, 0, 0, 1);
  return Transform2<T>(m, inv);
}

template <typename T> class Transform {
public:
  __host__ __device__ Transform() {
    m.setIdentity();
    m_inv.setIdentity();
  }
  __host__ __device__ Transform(const Matrix4x4<T> &mat)
      : m(mat), m_inv(inverse(mat)) {}
  __host__ __device__ Transform(const Matrix4x4<T> &mat,
                                const Matrix4x4<T> &inv_mat)
      : m(mat), m_inv(inv_mat) {}
  __host__ __device__ Transform(const T mat[4][4]) {
    m = Matrix4x4<T>(mat[0][0], mat[0][1], mat[0][2], mat[0][3], mat[1][0],
                     mat[1][1], mat[1][2], mat[1][3], mat[2][0], mat[2][1],
                     mat[2][2], mat[2][3], mat[3][0], mat[3][1], mat[3][2],
                     mat[3][3]);
    m_inv = inverse(m);
  }
  __host__ __device__ Transform(const bbox3 &bbox) {
    m.m[0][0] = bbox.upper[0] - bbox.lower[0];
    m.m[1][1] = bbox.upper[1] - bbox.lower[1];
    m.m[2][2] = bbox.upper[2] - bbox.lower[2];
    m.m[0][3] = bbox.lower[0];
    m.m[1][3] = bbox.lower[1];
    m.m[2][3] = bbox.lower[2];
    m_inv = inverse(m);
  }
  __host__ __device__ void reset() { m.setIdentity(); }
  __host__ __device__ void translate(const Vector3<T> &d) {
    // TODO update inverse and make a better implementarion
    UNUSED_VARIABLE(d);
  }
  __host__ __device__ void scale(T x, T y, T z) {
    // TODO update inverse and make a better implementarion
    UNUSED_VARIABLE(x);
    UNUSED_VARIABLE(y);
    UNUSED_VARIABLE(z);
  }
  template <typename S>
  friend __host__ __device__ Transform<S> inverse(const Transform<S> &t);
  __host__ __device__ bbox3 operator()(const bbox3 &b) const {
    const Transform &M = *this;
    bbox3 ret(M(Point3<T>(b.lower.x, b.lower.y, b.lower.z)));
    ret = make_union(ret, M(Point3<T>(b.upper.x, b.lower.y, b.lower.z)));
    ret = make_union(ret, M(Point3<T>(b.lower.x, b.upper.y, b.lower.z)));
    ret = make_union(ret, M(Point3<T>(b.lower.x, b.lower.y, b.upper.z)));
    ret = make_union(ret, M(Point3<T>(b.lower.x, b.upper.y, b.upper.z)));
    ret = make_union(ret, M(Point3<T>(b.upper.x, b.upper.y, b.lower.z)));
    ret = make_union(ret, M(Point3<T>(b.upper.x, b.lower.y, b.upper.z)));
    ret = make_union(ret, M(Point3<T>(b.lower.x, b.upper.y, b.upper.z)));
    return ret;
  }
  __host__ __device__ Point3<T> operator()(const Point2<T> &p) const {
    T x = p.x, y = p.y, z = 0.f;
    T xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
    T yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
    T zp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
    T wp = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];
    if (wp == 1.f)
      return Point3<T>(xp, yp, zp);
    return Point3<T>(xp, yp, zp) / wp;
  }
  __host__ __device__ Point3<T> operator()(const Point3<T> &p) const {
    T x = p.x, y = p.y, z = p.z;
    T xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
    T yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
    T zp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
    T wp = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];
    if (wp == 1.f)
      return Point3<T>(xp, yp, zp);
    return Point3<T>(xp, yp, zp) / wp;
  }
  __host__ __device__ void operator()(const Point3<T> &p, Point3<T> *r) const {
    T x = p.x, y = p.y, z = p.z;
    r->x = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
    r->y = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
    r->z = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
    T wp = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];
    if (wp != 1.f)
      *r /= wp;
  }
  __host__ __device__ Vector3<T> operator()(const Vector3<T> &v) const {
    T x = v.x, y = v.y, z = v.z;
    return Vector3<T>(m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z,
                      m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z,
                      m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z);
  }
  __host__ __device__ Transform &operator=(const Transform2<T> &t) {
    m.setIdentity();
    Matrix3x3<T> m3 = t.getMatrix();
    m.m[0][0] = m3.m[0][0];
    m.m[0][1] = m3.m[0][1];
    m.m[0][3] = m3.m[0][2];

    m.m[1][0] = m3.m[1][0];
    m.m[1][1] = m3.m[1][1];
    m.m[1][3] = m3.m[1][2];

    m_inv = inverse(m);
    return *this;
  }
  __host__ __device__ Transform operator*(const Transform &t) const {
    Matrix4x4<T> m1 = Matrix4x4<T>::mul(m, t.m);
    Matrix4x4<T> m1_inv = Matrix4x4<T>::mul(t.m_inv, m_inv);
    return Transform(m1, m1_inv);
  }
  __host__ __device__ Point3<T> operator*(const Point3<T> &p) const {
    return (*this)(p);
  }
  __host__ __device__ bool operator==(const Transform &t) const {
    return t.m == m;
  }
  __host__ __device__ bool operator!=(const Transform &t) const {
    return t.m != m;
  }
  /// \return true if this transformation changes the coordinate system
  /// handedness
  __host__ __device__ bool swapsHandedness() const {
    T det = (m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1])) -
            (m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0])) +
            (m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]));
    return det < 0;
  }
  __host__ __device__ const T *c_matrix() const { return &m.m[0][0]; }
  __host__ __device__ const Matrix4x4<T> &matrix() const { return m; }
  __host__ __device__ Matrix3x3<T> upperLeftMatrix() const {
    return Matrix3x3<T>(m.m[0][0], m.m[0][1], m.m[0][2], m.m[1][0], m.m[1][1],
                        m.m[1][2], m.m[2][0], m.m[2][1], m.m[2][2]);
  }
  __host__ __device__ Vector3<T> getTranslate() const {
    return Vector3<T>(m.m[0][3], m.m[1][3], m.m[2][3]);
  }
  __host__ __device__ void computeInverse() { m_inv = inverse(m); }
  __host__ __device__ bool isIdentity() { return m.isIdentity(); }
  __host__ __device__ void applyToPoint(const T *p, T *r, size_t d = 3) const {
    T x = p[0], y = p[1], z = 0.f;
    if (d == 3)
      z = p[2];
    r[0] = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
    r[1] = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
    if (d == 3)
      r[2] = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
    T wp = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];
    if (wp != 1.f) {
      T invwp = 1.f / wp;
      r[0] *= invwp;
      r[1] *= invwp;
      if (d == 3)
        r[2] *= invwp;
    }
  }

protected:
  Matrix4x4<T> m, m_inv;
};

template <typename T>
__host__ __device__ Transform<T>
segmentToSegmentTransform(Point3<T> a, Point3<T> b, Point3<T> c, Point3<T> d) {
  // Consider two bases a b e f and c d g h
  // TODO implement
  return Transform<T>();
}

template <typename T>
__host__ __device__ Transform<T> inverse(const Transform<T> &t) {
  return Transform<T>(t.m_inv, t.m);
}

template <typename T>
__host__ __device__ Transform<T> translate(const Vector3<T> &d) {
  Matrix4x4<T> m(1.f, 0.f, 0.f, d.x, 0.f, 1.f, 0.f, d.y, 0.f, 0.f, 1.f, d.z,
                 0.f, 0.f, 0.f, 1.f);
  Matrix4x4<T> m_inv(1.f, 0.f, 0.f, -d.x, 0.f, 1.f, 0.f, -d.y, 0.f, 0.f, 1.f,
                     -d.z, 0.f, 0.f, 0.f, 1.f);
  return Transform<T>(m, m_inv);
}

template <typename T> __host__ __device__ Transform<T> scale(T x, T y, T z) {
  Matrix4x4<T> m(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);
  Matrix4x4<T> inv(1.f / x, 0, 0, 0, 0, 1.f / y, 0, 0, 0, 0, 1.f / z, 0, 0, 0,
                   0, 1);
  return Transform<T>(m, inv);
}

template <typename T> __host__ __device__ Transform<T> rotateX(T angle) {
  T sin_a = sinf(TO_RADIANS(angle));
  T cos_a = cosf(TO_RADIANS(angle));
  Matrix4x4<T> m(1.f, 0.f, 0.f, 0.f, 0.f, cos_a, -sin_a, 0.f, 0.f, sin_a, cos_a,
                 0.f, 0.f, 0.f, 0.f, 1.f);
  return Transform<T>(m, transpose(m));
}

template <typename T> __host__ __device__ Transform<T> rotateY(T angle) {
  T sin_a = sinf(TO_RADIANS(angle));
  T cos_a = cosf(TO_RADIANS(angle));
  Matrix4x4<T> m(cos_a, 0.f, sin_a, 0.f, 0.f, 1.f, 0.f, 0.f, -sin_a, 0.f, cos_a,
                 0.f, 0.f, 0.f, 0.f, 1.f);
  return Transform<T>(m, transpose(m));
}

template <typename T> __host__ __device__ Transform<T> rotateZ(T angle) {
  T sin_a = sinf(TO_RADIANS(angle));
  T cos_a = cosf(TO_RADIANS(angle));
  Matrix4x4<T> m(cos_a, -sin_a, 0.f, 0.f, sin_a, cos_a, 0.f, 0.f, 0.f, 0.f, 1.f,
                 0.f, 0.f, 0.f, 0.f, 1.f);
  return Transform<T>(m, transpose(m));
}

template <typename T>
__host__ __device__ Transform<T> rotate(T angle, const Vector3<T> &axis) {
  Vector3<T> a = normalize(axis);
  T s = sinf(TO_RADIANS(angle));
  T c = cosf(TO_RADIANS(angle));
  T m[4][4];

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

  Matrix4x4<T> mat(m);
  return Transform<T>(mat, transpose(mat));
}
/*
template <typename T>
__host__ __device__ Transform<T> frustumTransform(T left, T right, T bottom,
                                                  T top, T near, T far) {
  T tnear = 2.f * near;
  T lr = right - left;
  T bt = top - bottom;
  T nf = far - near;
  T m[4][4];
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

  Matrix4x4<T> projection(m);
  return Transform<T>(projection, inverse(projection));
}

template <typename T>
__host__ __device__ Transform<T> perspective(T fov, T aspect, T zNear, T
zFar) { T xmax = zNear * tanf(TO_RADIANS(fov / 2.f)); T ymax = xmax /
aspect; return frustumTransform(-xmax, xmax, -ymax, ymax, zNear, zFar);
}

template <typename T>
__host__ __device__ Transform<T> perspective(T fov, T n, T f) {
  // perform projectiev divide
  Matrix4x4<T> persp = Matrix4x4<T>(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, f / (f -
n), -f * n / (f - n), 0, 0, 1, 0);
  // scale to canonical viewing volume
  T invTanAng = 1.f / tanf(TO_RADIANS(fov) / 2.f);
  return scale(invTanAng, invTanAng, 1) * Transform<T>(persp);
}

template <typename T>
__host__ __device__ Transform<T>
lookAt(const Point3<T> &pos, const Point3<T> &target, const Vector3<T> &up)
{ Vector3<T> dir = normalize(target - pos); Vector3<T> left =
normalize(cross(normalize(up), dir)); Vector3<T> new_up = cross(dir,
left); T m[4][4]; m[0][0] = left.x; m[1][0] = left.y; m[2][0] =
left.z; m[3][0] = 0;

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

  Matrix4x4<T> cam_to_world(m);
  return Transform<T>(inverse(cam_to_world), cam_to_world);
}

template <typename T>
__host__ __device__ Transform<T>
lookAtRH(const Point3<T> &pos, const Point3<T> &target, const Vector3<T>
&up) { Vector3<T> dir = normalize(pos - target); Vector3<T> left =
normalize(cross(normalize(up), dir)); Vector3<T> new_up = cross(dir,
left); T m[4][4]; m[0][0] = left.x; m[0][1] = left.y; m[0][2] =
left.z; m[0][3] = -dot(left, Vector3<T>(pos - Point3<T>()));

  m[1][0] = new_up.x;
  m[1][1] = new_up.y;
  m[1][2] = new_up.z;
  m[1][3] = -dot(new_up, Vector3<T>(pos - Point3<T>()));

  m[2][0] = dir.x;
  m[2][1] = dir.y;
  m[2][2] = dir.z;
  m[2][3] = -dot(dir, Vector3<T>(pos - Point3<T>()));

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
  */ /**
Matrix4x4<T> cam_to_world(m);
return Transform<T>(cam_to_world, inverse(cam_to_world));
} // namespace cuda

// Same as OpenGL convention
template <typename T>
__host__ __device__ Transform<T> ortho(T left, T right, T bottom, T top,
                                       T near = -1.f, T far = 1.f) {
  T m[4][4];

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

  Matrix4x4<T> projection(m);
  return Transform<T>(projection, inverse(projection));
}

template <typename T>
__host__ __device__ Transform<T> orthographic(T znear, T zfar) {
  return scale(1.f, 1.f, 1.f / (zfar - znear)) *
         translate(Vector3<T>(0.f, 0.f, -znear));
}
*/
using Transform2f = Transform2<float>;
using Transform3f = Transform<float>;

} // namespace cuda

} // namespace hermes

#endif // HERMES_GEOMETRY_CUDA_TRANSFORM_H
