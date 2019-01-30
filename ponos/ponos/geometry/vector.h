/*
 * Copyright (c) 2017 FilipeCN
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

#ifndef PONOS_GEOMETRY_VECTOR_H
#define PONOS_GEOMETRY_VECTOR_H

#include <ponos/geometry/normal.h>
#include <ponos/geometry/numeric.h>
#include <ponos/log/debug.h>

#include <cstring>
#include <initializer_list>
#include <vector>

namespace ponos {
template <typename T> class Point2;
template <typename T> class Vector2 {
public:
  Vector2();
  explicit Vector2(T _x, T _y);
  explicit Vector2(const Point2<T> &p);
  explicit Vector2(const Normal2<T> &n);
  explicit Vector2(T f);
  explicit Vector2(T *f);
  // access
  T operator[](size_t i) const;
  T &operator[](size_t i);
  // arithmetic
  Vector2 operator+(const Vector2 &v) const;
  Vector2 &operator+=(const Vector2 &v);
  Vector2 operator-(const Vector2 &v) const;
  Vector2 &operator-=(const Vector2 &v);
  Vector2 operator*(T f) const;
  Vector2 &operator*=(T f);
  Vector2 operator/(T f) const;
  Vector2 &operator/=(T f);
  Vector2 operator-() const;
  bool operator==(const Vector2 &v);
  // normalization
  T length2() const;
  T length() const;
  Vector2 right() const;
  Vector2 left() const;
  bool HasNaNs() const;
  friend std::ostream &operator<<(std::ostream &os, const Vector2 &v);
  T x, y;
};

template <typename T> Vector2<T> operator*(T f, const Vector2<T> &v);
template <typename T> Vector2<T> operator/(T f, const Vector2<T> &v);
template <typename T> T dot(const Vector2<T> &a, const Vector2<T> &b);

template <typename T>
inline T dot(const std::vector<T> &a, const std::vector<T> &b) {
  ASSERT_FATAL(a.size() == b.size());
  T sum = T(0);
  for (size_t i = 0; i < a.size(); i++)
    sum = sum + a[i] * b[i];
  return sum;
}

template <typename T> Vector2<T> normalize(const Vector2<T> &v);
template <typename T>
Vector2<T> orthonormal(const Vector2<T> &v, bool first = true);
template <typename T> T cross(const Vector2<T> &a, const Vector2<T> &b);

/** Projects a vector onto another.
 * \param a **[in]**
 * \param b **[in]**
 *
 * \returns the projection of **a** onto **b**
 */
template <typename T>
Vector2<T> project(const Vector2<T> &a, const Vector2<T> &b);

template <typename T> class Point3;
template <typename T> class Vector3 {
public:
  Vector3();
  explicit Vector3(T _f);
  explicit Vector3(T _x, T _y, T _z);
  explicit Vector3(const T *v);
  explicit Vector3(const Normal3<T> &n);
  explicit Vector3(const Point3<T> &p);
  // boolean
  bool operator==(const Vector3 &v) const;
  bool operator<(const Vector3 &v) const;
  bool operator>(const Vector3 &v) const;
  bool operator>=(const Vector3 &p) const;
  bool operator<=(const Vector3 &p) const;
  // access
  T operator[](int i) const;
  T &operator[](int i);
  Vector2<T> xy() const;
  // arithmetic
  Vector3 &operator=(const T &v);
  Vector3 operator+(const Vector3 &v) const;
  Vector3 &operator+=(const Vector3 &v);
  Vector3 operator-(const Vector3 &v) const;
  Vector3 &operator-=(const Vector3 &v);
  Vector3 operator*(const Vector3 &v) const;
  Vector3 operator*(T f) const;
  Vector3 &operator*=(T f);
  Vector3 operator/(T f) const;
  Vector3 &operator/=(T f);
  Vector3 &operator/=(const Vector3 &v);
  Vector3 operator-() const;
  // normalization
  T length2() const;
  T length() const;
  bool HasNaNs() const;
  friend std::ostream &operator<<(std::ostream &os, const Vector3 &v);
  T x, y, z;
};

template <typename T> Vector3<T> operator*(T f, const Vector3<T> &v);
template <typename T> T dot(const Vector3<T> &a, const Vector3<T> &b);
template <typename T>
Vector3<T> cross(const Vector3<T> &a, const Vector3<T> &b);
template <typename T>
T triple(const Vector3<T> &a, const Vector3<T> &b, const Vector3<T> &c);
template <typename T> Vector3<T> normalize(const Vector3<T> &v);
/** \brief compute the two orthogonal-tangential vectors from a
 * \param a **[in]** normal
 * \param b **[out]** first tangent
 * \param c **[out]** second tangent
 */
template <typename T>
void tangential(const Vector3<T> &a, Vector3<T> &b, Vector3<T> &c);
template <typename T> Vector3<T> cos(const Vector3<T> &v);
template <typename T> Vector3<T> max(const Vector3<T> &a, const Vector3<T> &b);
template <typename T> Vector3<T> abs(const Vector3<T> &a);


template <typename T> class Vector4 {
public:
  Vector4();
  explicit Vector4(T _x, T _y, T _z, T _w);
  // access
  T operator[](int i) const;
  T &operator[](int i);
  Vector2<T> xy();
  Vector3<T> xyz();
  // arithmetic
  Vector4 operator+(const Vector4 &v) const;
  Vector4 &operator+=(const Vector4 &v);
  Vector4 operator-(const Vector4 &v) const;
  Vector4 &operator-=(const Vector4 &v);
  Vector4 operator*(T f) const;
  Vector4 &operator*=(T f);
  Vector4 operator/(T f) const;
  Vector4 &operator/=(T f);
  Vector4 operator-() const;
  // normalization
  T length2() const;
  T length() const;
  bool HasNaNs() const;

  friend std::ostream &operator<<(std::ostream &os, const Vector4 &v);
  T x, y, z, w;
};

typedef Vector2<real_t> vec2;
typedef Vector3<real_t> vec3;
typedef Vector4<real_t> vec4;
typedef Vector3<double> vec3d;
typedef Vector3<float> vec3f;
typedef Vector2<float> vec2f;
typedef Vector2<int> ivec2;
typedef Vector2<uint> uivec2;
typedef Vector3<int> ivec3;
typedef Vector3<uint> uivec3;
typedef Vector4<int> ivec4;
typedef Vector4<uint> uivec4;

template <typename T = float, size_t D = 3> class Vector {
public:
  Vector();
  Vector(std::initializer_list<T> values);
  Vector(size_t n, const T *t);
  Vector(const T &t);
  Vector(const T &x, const T &y);
  Vector(const T &x, const T &y, const T &z);
  T operator[](int i) const;
  T &operator[](int i);
  bool operator==(const Vector<T, D> &_v) const;
  bool operator!=(const Vector<T, D> &_v) const;
  bool operator<=(const Vector<T, D> &_v) const;
  bool operator<(const Vector<T, D> &_v) const;
  bool operator>=(const Vector<T, D> &_v) const;
  bool operator>(const Vector<T, D> &_v) const;
  Vector<T, D> operator-(const Vector<T, D> &_v) const;
  Vector<T, D> operator+(const Vector<T, D> &_v) const;
  Vector<T, D> operator*(const Vector<T, D> &_v) const;
  Vector<T, D> operator/(const Vector<T, D> &_v) const;
  Vector<T, D> operator/=(T f);
  Vector<T, D> operator-=(const Vector<T, D> &_v);
  Vector<T, 2> operator/(T f) const;
  Vector<T, 2> xy(size_t x = 0, size_t y = 1) const;
  Vector<T, 2> floatXY(size_t x = 0, size_t y = 1) const;
  Vector<float, 3> floatXYZ(size_t x = 0, size_t y = 1, size_t z = 2);
  T max() const;
  T length2() const;
  T length() const;
  Vector<T, D> normalized() const;
  Vector<T, 2> right() const;
  Vector<T, 2> left() const;
  friend std::ostream &operator<<(std::ostream &os, const Vector &v) {
    os << "Vector[<" << D << ">]";
    for (size_t i = 0; i < v.size; i++)
      os << v[i] << " ";
    os << std::endl;
    return os;
  }

  size_t size;
  T v[D];
};

#include "vector.inl"

template <typename T, size_t D>
inline Vector<T, D> operator*(T f, const Vector<T, D> &v) {
  return v * f;
}

/* round
 * @v **[in]** vector
 * @return a vector with ceil applied to all components
 */
template <typename T> Vector<int, 3> ceil(const Vector3<T> &v);
/* round
 * @v **[in]** vector
 * @return a vector with floor applied to all components
 */
template <typename T> Vector<int, 3> floor(const Vector3<T> &v);

template <typename T> Vector<int, 3> min(Vector<int, 3> a, Vector<int, 3> b);
template <typename T> Vector<int, 3> max(Vector<int, 3> a, Vector<int, 3> b);

/* round
 * @v **[in]** vector
 * @return a vector with ceil applied to all components
 */
template <typename T> Vector<int, 2> ceil(const Vector2<T> &v);
/* round
 * @v **[in]** vector
 * @return a vector with floor applied to all components
 */
template <typename T> Vector<int, 2> floor(const Vector2<T> &v);

Vector<int, 2> min(Vector<int, 2> a, Vector<int, 2> b);
Vector<int, 2> max(Vector<int, 2> a, Vector<int, 2> b);

template <typename T, size_t D> T normalize(Vector<T, D> &v) {
  T d = v.length();
  for (size_t i = 0; i < D; i++)
    v[i] = v[i] / d;
}

} // namespace ponos

#endif
