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

#ifndef HERMES_GEOMETRY_CUDA_POINT_H
#define HERMES_GEOMETRY_CUDA_POINT_H

#include <hermes/geometry/vector.h>
#include <ponos/geometry/point.h>

namespace hermes {

namespace cuda {

template <typename T> class Point2 {
public:
  typedef T ScalarType;
  __host__ __device__ Point2();
  __host__ __device__ Point2(T f);
  __host__ __device__ Point2(T _x, T _y);
  __host__ Point2(const ponos::Point2<T> &ponos_point)
      : x(ponos_point.x), y(ponos_point.y) {}
  // access
  __host__ __device__ T operator[](int i) const;
  __host__ __device__ T &operator[](int i);
  __host__ __device__ bool operator==(const Point2 &p) const;
  __host__ __device__ Point2 operator+(const Vector2<T> &v) const;
  __host__ __device__ Point2 operator-(const Vector2<T> &v) const;
  __host__ __device__ Point2 operator-(const T &f) const;
  __host__ __device__ Point2 operator+(const T &f) const;
  __host__ __device__ Vector2<T> operator-(const Point2 &p) const;
  __host__ __device__ Point2 operator/(T d) const;
  __host__ __device__ Point2 operator*(T f) const;
  __host__ __device__ Point2 &operator+=(const Vector2<T> &v);
  __host__ __device__ Point2 &operator-=(const Vector2<T> &v);
  __host__ __device__ Point2 &operator/=(T d);
  __host__ __device__ bool operator<(const Point2 &p) const;
  __host__ __device__ bool operator>=(const Point2 &p) const;
  __host__ __device__ bool operator<=(const Point2 &p) const;
  bool HasNaNs() const;

  ponos::Point2<T> ponos() const { return ponos::Point2<T>(x, y); }
  template <typename TT>
  friend std::ostream &operator<<(std::ostream &os, const Point2<TT> &p);
  T x, y;
};

typedef Point2<float> point2;
typedef Point2<unsigned int> point2u;
typedef Point2<int> point2i;
typedef Point2<float> point2f;
typedef Point2<double> point2d;

template <typename T>
__host__ __device__ Point2<T> operator*(T f, const Point2<T> &p);
template <typename T>
__host__ __device__ T distance(const Point2<T> &a, const Point2<T> &b);
template <typename T>
__host__ __device__ T distance2(const Point2<T> &a, const Point2<T> &b);
template <typename T> __host__ __device__ Point2<T> floor(const Point2<T> &a) {
  return {static_cast<int>(a.x), static_cast<int>(a.y)};
}

template <typename T> class Point3 {
public:
  __host__ __device__ Point3() {}
  __host__ __device__ Point3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
  __host__ __device__ Point3(const Vector3<T> &v) : x(v.x), y(v.y), z(v.z) {}
  __host__ __device__ Point3(const Point2<T> &p) : x(p.x), y(p.y), z(0) {}
  __host__ __device__ Point3(const T *v) : x(v[0]), y(v[1]), z(v[2]) {}
  __host__ __device__ Point3(T v) : x(v), y(v), z(v) {}
  // access
  __host__ __device__ T operator[](int i) const { return (&x)[i]; }
  __host__ __device__ T &operator[](int i) { return (&x)[i]; }
  __host__ __device__ Point2<T> xy() const { return Point2<T>(x, y); }
  __host__ __device__ Point2<T> yz() const { return Point2<T>(y, z); }
  __host__ __device__ Point2<T> xz() const { return Point2<T>(x, z); }
  __host__ __device__ T u() const { return x; }
  __host__ __device__ T v() const { return y; }
  __host__ __device__ T s() const { return z; }
  // arithmetic
  __host__ __device__ Point3<T> operator+() const { return *this; }
  __host__ __device__ Point3<T> operator-() const {
    return Point3<T>(-x, -y, -z);
  }
  __host__ __device__ Point3<T> &operator+=(const Vector3<T> &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }
  __host__ __device__ Point3<T> &operator*=(T d) {
    x *= d;
    y *= d;
    z *= d;
    return *this;
  }
  __host__ __device__ Point3<T> &operator-=(const Vector3<T> &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }
  __host__ __device__ Point3<T> &operator/=(T d) {
    x /= d;
    y /= d;
    z /= d;
    return *this;
  }
  // boolean
  template <typename TT>
  friend std::ostream &operator<<(std::ostream &os, const Point3<TT> &p);
  T x = 0, y = 0, z = 0;
};

typedef Point3<float> point3;
typedef Point3<unsigned int> point3u;
typedef Point3<int> point3i;
typedef Point3<float> point3f;
typedef Point3<float> point3d;

template <typename T>
__host__ __device__ Point3<T> operator+(T f, const Point3<T> &p) {
  return Point3<T>(p.x + f, p.y + f, p.z + f);
}
template <typename T>
__host__ __device__ Point3<T> operator-(T f, const Point3<T> &p) {
  return Point3<T>(p.x - f, p.y - f, p.z - f);
}
template <typename T>
__host__ __device__ Point3<T> operator*(T d, const Point3<T> &p) {
  return Point3<T>(p.x * d, p.y * d, p.z * d);
}
template <typename T>
__host__ __device__ Point3<T> operator+(const Point3<T> &p, T f) {
  return Point3<T>(p.x + f, p.y + f, p.z + f);
}
template <typename T>
__host__ __device__ Point3<T> operator-(const Point3<T> &p, T f) {
  return Point3<T>(p.x - f, p.y - f, p.z - f);
}
template <typename T>
__host__ __device__ Point3<T> operator*(const Point3<T> &p, T d) {
  return Point3<T>(p.x * d, p.y * d, p.z * d);
}
template <typename T>
__host__ __device__ Point3<T> operator/(const Point3<T> &p, T d) {
  return Point3<T>(p.x / d, p.y / d, p.z / d);
}
template <typename T>
__host__ __device__ Point3<T> operator+(const Point3<T> &p,
                                        const Vector3<T> &v) {
  return Point3<T>(p.x + v.x, p.y + v.y, p.z + v.z);
}
template <typename T>
__host__ __device__ Point3<T> operator-(const Point3<T> &p,
                                        const Vector3<T> &v) {
  return Point3<T>(p.x - v.x, p.y - v.y, p.z - v.z);
}
template <typename T>
__host__ __device__ Point3<T> operator+(const Point3<T> &p,
                                        const Point3<T> &v) {
  return Point3<T>(p.x + v.x, p.y + v.y, p.z + v.z);
}
template <typename T>
__host__ __device__ Vector3<T> operator-(const Point3<T> &q,
                                         const Point3<T> &p) {
  return Vector3<T>(q.x - p.x, q.y - p.y, q.z - p.z);
}
template <typename T>
__host__ __device__ bool operator==(const Point3<T> &p, const Point3<T> &q) {
  return Check::is_equal(p.x, q.x) && Check::is_equal(p.y, q.y) &&
         Check::is_equal(p.z, q.z);
}
template <typename T>
__host__ __device__ bool operator>=(const Point3<T> &q, const Point3<T> &p) {
  return q.x >= p.x && q.y >= p.y && q.z >= p.z;
}
template <typename T>
__host__ __device__ bool operator<=(const Point3<T> &q, const Point3<T> &p) {
  return q.x <= p.x && q.y <= p.y && q.z <= p.z;
}
template <typename T>
__host__ __device__ bool operator<(const Point3<T> &left,
                                   const Point3<T> &right) {
  if (left.x < right.x)
    return true;
  else if (left.x > right.x)
    return false;
  if (left.y < right.y)
    return true;
  else if (left.y > right.y)
    return false;
  if (left.z < right.z)
    return true;
  else if (left.z > right.z)
    return false;
  return false;
}

template <typename T>
T __host__ __device__ distance(const Point3<T> &a, const Point3<T> &b) {
  return (a - b).length();
}
template <typename T>
T __host__ __device__ distance2(const Point3<T> &a, const Point3<T> &b) {
  return (a - b).length2();
}
template <typename T> __host__ __device__ Point3<T> floor(const Point3<T> &a) {
  return {static_cast<int>(a.x), static_cast<int>(a.y), static_cast<int>(a.z)};
}

#include "cuda_point.inl"

} // namespace cuda

} // namespace hermes

#endif
