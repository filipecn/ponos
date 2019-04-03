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

#include <hermes/geometry/cuda_vector.h>

namespace hermes {

namespace cuda {

template <typename T> class Point2 {
public:
  typedef T ScalarType;
  __host__ __device__ Point2();
  __host__ __device__ Point2(T f);
  __host__ __device__ explicit Point2(const T *v);
  __host__ __device__ explicit Point2(T _x, T _y);
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
  template <typename TT>
  friend std::ostream &operator<<(std::ostream &os, const Point2<TT> &p);
  T x, y;
};

typedef Point2<float> point2;
typedef Point2<uint> point2u;
typedef Point2<int> point2i;
typedef Point2<float> point2f;
typedef Point2<double> point2d;

template <typename T>
__host__ __device__ Point2<T> operator*(T f, const Point2<T> &p);
template <typename T>
__host__ __device__ T distance(const Point2<T> &a, const Point2<T> &b);
template <typename T>
__host__ __device__ T distance2(const Point2<T> &a, const Point2<T> &b);

template <typename T> class Point3 {
public:
  __host__ __device__ Point3();
  __host__ __device__ explicit Point3(T _x, T _y, T _z);
  __host__ __device__ explicit Point3(const Vector3<T> &v);
  __host__ __device__ explicit Point3(const T *v);
  __host__ __device__ explicit Point3(T v);
  __host__ __device__ explicit Point3(const Point2<T> &p);
  __host__ __device__ explicit operator Vector3<T>() const {
    return Vector3<T>(x, y, z);
  }
  // access
  __host__ __device__ T operator[](int i) const;
  __host__ __device__ T &operator[](int i);
  // arithmetic
  __host__ __device__ Point3 operator+(const Vector3<T> &v) const;
  __host__ __device__ Point3 operator+(const Point3<T> &v) const;
  __host__ __device__ Point3 operator+(const T &f) const;
  __host__ __device__ Point3 operator-(const T &f) const;
  __host__ __device__ Point3 &operator+=(const Vector3<T> &v);
  __host__ __device__ Vector3<T> operator-(const Point3 &p) const;
  __host__ __device__ Point3 operator-(const Vector3<T> &v) const;
  __host__ __device__ Point3 &operator-=(const Vector3<T> &v);
  __host__ __device__ bool operator==(const Point3 &p) const;
  __host__ __device__ bool operator>=(const Point3 &p) const;
  __host__ __device__ bool operator<=(const Point3 &p) const;
  __host__ __device__ Point3 operator*(T d) const;
  __host__ __device__ Point3 operator/(T d) const;
  __host__ __device__ Point3 &operator/=(T d);
  __host__ __device__ Point3 &operator*=(T d);
  // boolean
  __host__ __device__ bool operator==(const Point3 &p);
  __host__ __device__ Point2<T> xy() const;
  __host__ __device__ Point2<T> yz() const;
  __host__ __device__ Point2<T> xz() const;
  __host__ __device__ Vector3<T> asVector3() const;
  __host__ __device__ vec3i asIVec3() const;
  bool HasNaNs() const;
  static uint dimension() { return 3; }
  template <typename TT>
  friend std::ostream &operator<<(std::ostream &os, const Point3<TT> &p);
  T x, y, z;
};

typedef Point3<float> point3;
typedef Point3<uint> point3u;
typedef Point3<int> point3i;
typedef Point3<float> point3f;
typedef Point3<float> point3d;

template <typename T>
__host__ __device__ T distance(const Point3<T> &a, const Point3<T> &b);
template <typename T>
__host__ __device__ T distance2(const Point3<T> &a, const Point3<T> &b);

#include "cuda_point.inl"

} // namespace cuda

} // namespace hermes

#endif
