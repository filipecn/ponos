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
  __host__ bool HasNaNs() const;
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
  __host__ __device__ Vector3();
  __host__ __device__ explicit Vector3(T _f);
  __host__ __device__ explicit Vector3(T _x, T _y, T _z);
  __host__ __device__ explicit Vector3(const T *v);
  __host__ __device__ explicit Vector3(const Point3<T> &p);
  // boolean
  __host__ __device__ bool operator==(const Vector3 &v) const;
  __host__ __device__ bool operator<(const Vector3 &v) const;
  __host__ __device__ bool operator>(const Vector3 &v) const;
  __host__ __device__ bool operator>=(const Vector3 &p) const;
  __host__ __device__ bool operator<=(const Vector3 &p) const;
  // access
  __host__ __device__ T operator[](int i) const;
  __host__ __device__ T &operator[](int i);
  __host__ __device__ Vector2<T> xy(int i = 0, int j = 1) const;
  // arithmetic
  __host__ __device__ Vector3 &operator=(const T &v);
  __host__ __device__ Vector3 operator+(const Vector3 &v) const;
  __host__ __device__ Vector3 &operator+=(const Vector3 &v);
  __host__ __device__ Vector3 operator-(const Vector3 &v) const;
  __host__ __device__ Vector3 &operator-=(const Vector3 &v);
  __host__ __device__ Vector3 operator*(const Vector3 &v) const;
  __host__ __device__ Vector3 operator*(T f) const;
  __host__ __device__ Vector3 &operator*=(T f);
  __host__ __device__ Vector3 operator/(T f) const;
  __host__ __device__ Vector3 &operator/=(T f);
  __host__ __device__ Vector3 &operator/=(const Vector3 &v);
  __host__ __device__ Vector3 operator-() const;
  // normalization
  __host__ __device__ T length2() const;
  __host__ __device__ T length() const;
  __host__ __device__ bool HasNaNs() const;
  template <typename TT>
  friend std::ostream &operator<<(std::ostream &os, const Vector3<TT> &v);
  T x, y, z;
};

template <typename T>
__host__ __device__ Vector3<T> operator*(T f, const Vector3<T> &v);
template <typename T>
__host__ __device__ T dot(const Vector3<T> &a, const Vector3<T> &b);
template <typename T>
__host__ __device__ Vector3<T> cross(const Vector3<T> &a, const Vector3<T> &b);
template <typename T>
__host__ __device__ T triple(const Vector3<T> &a, const Vector3<T> &b,
                             const Vector3<T> &c);
template <typename T>
__host__ __device__ Vector3<T> normalize(const Vector3<T> &v);
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

// template <typename T, int D> class Vector {
// public:
//   __host__ __device__ Vector();
//   __host__ __device__ Vector(std::initializer_list<T> values);
//   __host__ __device__ Vector(const T &t);
//   __host__ __device__ Vector(const T &x, const T &y);
//   __host__ __device__ Vector(const T &x, const T &y, const T &z);
//   __host__ __device__ T operator[](int i) const;
//   __host__ __device__ T &operator[](int i);
//   __host__ __device__ bool operator==(const Vector<T, D> &_v) const;
//   __host__ __device__ bool operator!=(const Vector<T, D> &_v) const;
//   __host__ __device__ bool operator<=(const Vector<T, D> &_v) const;
//   __host__ __device__ bool operator<(const Vector<T, D> &_v) const;
//   __host__ __device__ bool operator>=(const Vector<T, D> &_v) const;
//   __host__ __device__ bool operator>(const Vector<T, D> &_v) const;
//   __host__ __device__ Vector<T, D> operator-(const Vector<T, D> &_v) const;
//   __host__ __device__ Vector<T, D> operator+(const Vector<T, D> &_v) const;
//   __host__ __device__ Vector<T, D> operator*(const Vector<T, D> &_v) const;
//   __host__ __device__ Vector<T, D> operator/(const Vector<T, D> &_v) const;
//   __host__ __device__ Vector<T, D> operator/=(T f);
//   __host__ __device__ Vector<T, D> operator-=(const Vector<T, D> &_v);
//   __host__ __device__ Vector<T, 2> operator/(T f) const;
//   __host__ __device__ Vector<T, 2> xy(size_t x = 0, size_t y = 1) const;
//   __host__ __device__ Vector<T, 2> floatXY(size_t x = 0, size_t y = 1) const;
//   __host__ __device__ Vector<float, 3> floatXYZ(size_t x = 0, size_t y = 1,
//                                                 size_t z = 2);
//   __host__ __device__ T max() const;
//   __host__ __device__ T length2() const;
//   __host__ __device__ T length() const;
//   __host__ __device__ Vector<T, D> normalized() const;
//   __host__ __device__ Vector<T, 2> right() const;
//   __host__ __device__ Vector<T, 2> left() const;
//   friend std::ostream &operator<<(std::ostream &os, const Vector &v) {
//     os << "Vector[<" << D << ">]";
//     for (size_t i = 0; i < v.size; i++)
//       os << v[i] << " ";
//     os << std::endl;
//     return os;
//   }

//   int size;
//   T v[D];
// };

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
// typedef Vector<unsigned int, 2> Vector2u;
// typedef Vector<unsigned int, 3> Vector3u;
// typedef Vector<float, 2> Vector2f;
// typedef Vector<float, 3> Vector3f;
// typedef Vector<double, 2> Vector2d;
// typedef Vector<double, 3> Vector3d;

#include "cuda_vector.inl"

} // namespace cuda

} // namespace hermes

#endif // HERMES_CUDA_VECTOR_H