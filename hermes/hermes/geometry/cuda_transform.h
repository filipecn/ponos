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
#include <hermes/geometry/cuda_bbox.h>
#include <hermes/geometry/cuda_matrix.h>
#include <hermes/geometry/cuda_vector.h>

namespace hermes {

namespace cuda {

template <typename T> class Transform2 {
public:
  __host__ __device__ Transform2();
  __host__ __device__ Transform2(const Matrix3x3<T> &mat,
                                 const Matrix3x3<T> inv_mat);
  __host__ __device__ Transform2(const BBox2<T> &bbox);
  __host__ __device__ void reset();
  __host__ __device__ void translate(const Vector2<T> &d);
  __host__ __device__ void scale(T x, T y);
  __host__ __device__ void rotate(T angle);
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

template <typename T> __host__ __device__ Transform2<T> scale(T x, T y);
template <typename T> __host__ __device__ Transform2<T> rotate(T angle);
template <typename T>
__host__ __device__ Transform2<T> translate(const Vector2<T> &v);
template <typename T>
__host__ __device__ Transform2<T> inverse(const Transform2<T> &t);

template <typename T> class Transform {
public:
  __host__ __device__ Transform();
  __host__ __device__ Transform(const Matrix4x4<T> &mat);
  __host__ __device__ Transform(const Matrix4x4<T> &mat,
                                const Matrix4x4<T> &inv_mat);
  __host__ __device__ Transform(const T mat[4][4]);
  __host__ __device__ Transform(const bbox3 &bbox);
  __host__ __device__ void reset();
  __host__ __device__ void translate(const Vector3<T> &d);
  __host__ __device__ void scale(T x, T y, T z);
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
  __host__ __device__ bool swapsHandedness() const;
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
segmentToSegmentTransform(Point3<T> a, Point3<T> b, Point3<T> c, Point3<T> d);
template <typename T>
__host__ __device__ Transform<T> inverse(const Transform<T> &t);
template <typename T>
__host__ __device__ Transform<T> translate(const Vector3<T> &d);
template <typename T> __host__ __device__ Transform<T> scale(T x, T y, T z);
template <typename T> __host__ __device__ Transform<T> rotateX(T angle);
template <typename T> __host__ __device__ Transform<T> rotateY(T angle);
template <typename T> __host__ __device__ Transform<T> rotateZ(T angle);
template <typename T>
__host__ __device__ Transform<T> rotate(T angle, const Vector3<T> &axis);
// Same as OpenGL convention
template <typename T>
__host__ __device__ Transform<T> frustumTransform(T left, T right, T bottom,
                                                  T top, T near, T far);
template <typename T>
__host__ __device__ Transform<T> perspective(T fovy, T aspect, T zNear, T zFar);
template <typename T>
__host__ __device__ Transform<T> perspective(T fovy, T zNear, T zFar);
template <typename T>
__host__ __device__ Transform<T>
lookAt(const Point3<T> &pos, const Point3<T> &target, const Vector3<T> &up);
template <typename T>
__host__ __device__ Transform<T>
lookAtRH(const Point3<T> &pos, const Point3<T> &target, const Vector3<T> &up);
template <typename T>
__host__ __device__ Transform<T> ortho(T left, T right, T bottom, T top,
                                       T near = -1.f, T far = 1.f);

template <typename T>
__host__ __device__ Transform<T> orthographic(T znear, T zfar);

#include "cuda_transform.inl"

} // namespace cuda

} // namespace hermes

#endif // HERMES_GEOMETRY_CUDA_TRANSFORM_H
