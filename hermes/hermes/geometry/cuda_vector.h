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

#ifndef HERMES_GEOMETRY_CUDA_VECTOR_H
#define HERMES_GEOMETRY_CUDA_VECTOR_H

#include <cuda_runtime.h>
#include <hermes/common/defs.h>
#include <iostream>

namespace hermes {

namespace cuda {

template <typename T> class Point2;
template <typename T> class Vector2 {
public:
  __host__ __device__ Vector2();
  __host__ __device__ explicit Vector2(T _x, T _y);
  __host__ __device__ explicit Vector2(const Point2<T> &p);
  __host__ __device__ explicit Vector2(T f);
  __host__ __device__ explicit Vector2(T *f);
  // access
  __host__ __device__ T operator[](size_t i) const;
  __host__ __device__ T &operator[](size_t i);
  // arithmetic
  __host__ __device__ Vector2 operator+(const Vector2 &v) const;
  __host__ __device__ Vector2 &operator+=(const Vector2 &v);
  __host__ __device__ Vector2 operator-(const Vector2 &v) const;
  __host__ __device__ Vector2 &operator-=(const Vector2 &v);
  __host__ __device__ Vector2 operator*(T f) const;
  __host__ __device__ Vector2 &operator*=(T f);
  __host__ __device__ Vector2 operator/(T f) const;
  __host__ __device__ Vector2 &operator/=(T f);
  __host__ __device__ Vector2 operator-() const;
  __host__ __device__ bool operator==(const Vector2 &v);
  // normalization
  __host__ __device__ T length2() const;
  __host__ __device__ T length() const;
  __host__ __device__ Vector2 right() const;
  __host__ __device__ Vector2 left() const;
  template <typename TT>
  friend std::ostream &operator<<(std::ostream &os, const Vector2<TT> &v);
  T x, y;
};

template <typename T>
__host__ __device__ Vector2<T> operator*(T f, const Vector2<T> &v);
template <typename T>
__host__ __device__ Vector2<T> operator/(T f, const Vector2<T> &v);
template <typename T>
__host__ __device__ T dot(const Vector2<T> &a, const Vector2<T> &b);
template <typename T>
__host__ __device__ Vector2<T> normalize(const Vector2<T> &v);
template <typename T>
__host__ __device__ Vector2<T> orthonormal(const Vector2<T> &v,
                                           bool first = true);
template <typename T>
__host__ __device__ T cross(const Vector2<T> &a, const Vector2<T> &b);
///  Projects a vector onto another.
/// \param a **[in]**
/// \param b **[in]**
/// \returns the projection of **a** onto **b**
template <typename T>
Vector2<T> project(const Vector2<T> &a, const Vector2<T> &b);

template <typename T> class Point3;
template <typename T> class Vector3 {
public:
  __host__ __device__ Vector3() { x = y = z = 0; }
  __host__ __device__ Vector3(T _f) : x(_f), y(_f), z(_f) {}
  __host__ __device__ Vector3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
  __host__ __device__ Vector3(const T *v) {
    x = v[0];
    y = v[1];
    z = v[2];
  }
  __host__ __device__ Vector3(const Point3<T> &p) : x(p.x), y(p.y), z(p.z) {}
  // access
  __host__ __device__ T operator[](int i) const { return (&x)[i]; }
  __host__ __device__ T &operator[](int i) { return (&x)[i]; }
  __host__ __device__ Vector2<T> xy() const { return Vector2<T>(x, y); }
  __host__ __device__ Vector2<T> xz() const { return Vector2<T>(x, z); }
  __host__ __device__ Vector2<T> yz() const { return Vector2<T>(y, z); }
  __host__ __device__ T r() const { return x; }
  __host__ __device__ T g() const { return y; }
  __host__ __device__ T b() const { return z; }
  // arithmetic
  __host__ __device__ Vector3<T> operator+() const { return *this; }
  __host__ __device__ Vector3<T> operator-() const {
    return Vector3(-x, -y, -z);
  }
  __host__ __device__ Vector3<T> &operator=(const T &v) {
    x = y = z = v;
    return *this;
  }
  __host__ __device__ Vector3<T> &operator-=(const Vector3<T> &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }
  __host__ __device__ Vector3<T> &operator+=(const Vector3<T> &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }
  __host__ __device__ Vector3<T> &operator*=(T f) {
    x *= f;
    y *= f;
    z *= f;
    return *this;
  }
  __host__ __device__ Vector3<T> &operator/=(T f) {
    Check::isEqual(f, 0.f);
    T inv = 1.f / f;
    x *= inv;
    y *= inv;
    z *= inv;
    return *this;
  }
  __host__ __device__ Vector3<T> &operator/=(const Vector3<T> &v) {
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
  }
  // normalization
  __host__ __device__ T length2() const { return x * x + y * y + z * z; }
  __host__ __device__ T length() const { return sqrtf(length2()); }

  T x, y, z;
};

template <typename T>
__host__ __device__ Vector3<T> operator-(const Vector3<T> &u,
                                         const Vector3<T> &v) {
  return Vector3<T>(u.x - v.x, u.y - v.y, u.z - v.z);
}
template <typename T>
__host__ __device__ Vector3<T> operator+(const Vector3<T> &u,
                                         const Vector3<T> &v) {
  return Vector3<T>(u.x + v.x, u.y + v.y, u.z + v.z);
}
template <typename T>
__host__ __device__ Vector3<T> operator*(const Vector3<T> &u,
                                         const Vector3<T> &v) {
  return Vector3<T>(u.x * v.x, u.y * v.y, u.z * v.z);
}
template <typename T>
__host__ __device__ Vector3<T> operator*(const Vector3<T> &v, T f) {
  return Vector3<T>(v.x * f, v.y * f, v.z * f);
}
template <typename T>
__host__ __device__ Vector3<T> operator*(T f, const Vector3<T> &v) {
  return Vector3<T>(v.x * f, v.y * f, v.z * f);
}
template <typename T>
__host__ __device__ Vector3<T> operator/(const Vector3<T> &v, T f) {
  Check::isEqual(f, 0.f);
  T inv = 1. / f;
  return Vector3<T>(v.x * inv, v.y * inv, v.z * inv);
}
// boolean
template <typename T>
__host__ __device__ bool operator>=(const Vector3<T> &v, const Vector3<T> &p) {
  return v.x >= p.x && v.y >= p.y && v.z >= p.z;
}
template <typename T>
__host__ __device__ bool operator<=(const Vector3<T> &v, const Vector3<T> &p) {
  return v.x <= p.x && v.y <= p.y && v.z <= p.z;
}
template <typename T>
__host__ __device__ bool operator<(const Vector3<T> &u, const Vector3<T> &v) {
  if (u.x < v.x)
    return true;
  if (u.y < v.y)
    return true;
  return u.z < v.z;
}
template <typename T>
__host__ __device__ bool operator>(const Vector3<T> &u, const Vector3<T> &v) {
  if (u.x > v.x)
    return true;
  if (u.y > v.y)
    return true;
  return u.z > v.z;
}
template <typename T>
__host__ __device__ bool operator==(const Vector3<T> &u, const Vector3<T> &v) {
  return Check::isEqual(u.x, v.x) && Check::isEqual(u.y, v.y) &&
         Check::isEqual(u.z, v.z);
}
// algebra
template <typename T>
__host__ __device__ T dot(const Vector3<T> &a, const Vector3<T> &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename T>
__host__ __device__ Vector3<T> cross(const Vector3<T> &a, const Vector3<T> &b) {
  return Vector3<T>((a.y * b.z) - (a.z * b.y), (a.z * b.x) - (a.x * b.z),
                    (a.x * b.y) - (a.y * b.x));
}

template <typename T>
__host__ __device__ T triple(const Vector3<T> &a, const Vector3<T> &b,
                             const Vector3<T> &c) {
  return dot(a, cross(b, c));
}

template <typename T>
__host__ __device__ Vector3<T> normalize(const Vector3<T> &v) {
  if (v.length2() == 0.f)
    return v;
  return v / v.length();
}
/// Compute the two orthogonal-tangential vectors from a
/// \param a **[in]** normal
/// \param b **[out]** first tangent
/// \param c **[out]** second tangent
template <typename T>
__host__ __device__ void tangential(const Vector3<T> &a, Vector3<T> &b,
                                    Vector3<T> &c);
template <typename T> __host__ __device__ Vector3<T> cos(const Vector3<T> &v);
template <typename T>
__host__ __device__ Vector3<T> max(const Vector3<T> &a, const Vector3<T> &b);
template <typename T> __host__ __device__ Vector3<T> abs(const Vector3<T> &a);

template <typename T>
__host__ std::ostream &operator<<(std::ostream &os, const Vector3<T> &v) {
  os << "[vector3]" << v.x << " " << v.y << " " << v.z << std::endl;
  return os;
}

typedef Vector2<double> vec2d;
typedef Vector3<double> vec3d;
typedef Vector3<float> vec3;
typedef Vector2<float> vec2;
typedef Vector3<float> vec3f;
typedef Vector2<float> vec2f;
typedef Vector2<int> vec2i;
typedef Vector2<uint> vec2u;
typedef Vector3<int> vec3i;
typedef Vector3<uint> vec3u;

#include "cuda_vector.inl"

} // namespace cuda

} // namespace hermes

#endif // HERMES_CUDA_VECTOR_H