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

class Point2;
class Vector2 {
public:
  Vector2();
  explicit Vector2(real_t _x, real_t _y);
  explicit Vector2(const Point2 &p);
  explicit Vector2(const Normal2D &n);
  explicit Vector2(real_t f);
  explicit Vector2(real_t *f);
  // access
  real_t operator[](int i) const;
  real_t &operator[](int i);
  // arithmetic
  Vector2 operator+(const Vector2 &v) const;
  Vector2 &operator+=(const Vector2 &v);
  Vector2 operator-(const Vector2 &v) const;
  Vector2 &operator-=(const Vector2 &v);
  Vector2 operator*(real_t f) const;
  Vector2 &operator*=(real_t f);
  Vector2 operator/(real_t f) const;
  Vector2 &operator/=(real_t f);
  Vector2 operator-() const;
  bool operator==(const Vector2 &v);
  // normalization
  real_t length2() const;
  real_t length() const;
  Vector2 right() const;
  Vector2 left() const;
  bool HasNaNs() const;
  friend std::ostream &operator<<(std::ostream &os, const Vector2 &v);
  real_t x, y;
};

Vector2 operator*(real_t f, const Vector2 &v);
Vector2 operator/(real_t f, const Vector2 &v);
real_t dot(const Vector2 &a, const Vector2 &b);

template <typename T>
inline T dot(const std::vector<T> &a, const std::vector<T> &b) {
  ASSERT_FATAL(a.size() == b.size());
  T sum = T(0);
  for (size_t i = 0; i < a.size(); i++)
    sum = sum + a[i] * b[i];
  return sum;
}

Vector2 normalize(const Vector2 &v);
Vector2 orthonormal(const Vector2 &v, bool first = true);
real_t cross(const Vector2 &a, const Vector2 &b);

/** Projects a vector onto another.
 * \param a **[in]**
 * \param b **[in]**
 *
 * \returns the projection of **a** onto **b**
 */
Vector2 project(const Vector2 &a, const Vector2 &b);

class Point3;
class Vector3 {
public:
  Vector3();
  explicit Vector3(real_t _f);
  explicit Vector3(real_t _x, real_t _y, real_t _z);
  explicit Vector3(const real_t *v);
  explicit Vector3(const Normal &n);
  explicit Vector3(const Point3 &p);
  // boolean
  bool operator==(const Vector3 &v);
  bool operator<(const Vector3 &v);
  bool operator>(const Vector3 &v);
  bool operator>=(const Vector3 &p) const;
  bool operator<=(const Vector3 &p) const;
  // access
  real_t operator[](int i) const;
  real_t &operator[](int i);
  Vector2 xy();
  // arithmetic
  Vector3 &operator=(const real_t &v);
  Vector3 operator+(const Vector3 &v) const;
  Vector3 &operator+=(const Vector3 &v);
  Vector3 operator-(const Vector3 &v) const;
  Vector3 &operator-=(const Vector3 &v);
  Vector3 operator*(const Vector3 &v) const;
  Vector3 operator*(real_t f) const;
  Vector3 &operator*=(real_t f);
  Vector3 operator/(real_t f) const;
  Vector3 &operator/=(real_t f);
  Vector3 &operator/=(const Vector3 &v);
  Vector3 operator-() const;
  // normalization
  real_t length2() const;
  real_t length() const;
  bool HasNaNs() const;
  friend std::ostream &operator<<(std::ostream &os, const Vector3 &v);
  real_t x, y, z;
};

Vector3 operator*(real_t f, const Vector3 &v);
real_t dot(const Vector3 &a, const Vector3 &b);
Vector3 cross(const Vector3 &a, const Vector3 &b);
real_t triple(const Vector3 &a, const Vector3 &b, const Vector3 &c);
Vector3 normalize(const Vector3 &v);
/** \brief compute the two orthogonal-tangential vectors from a
 * \param a **[in]** normal
 * \param b **[out]** first tangent
 * \param c **[out]** second tangent
 */
void tangential(const Vector3 &a, Vector3 &b, Vector3 &c);
Vector3 cos(const Vector3 &v);
Vector3 max(const Vector3 &a, const Vector3 &b);

class Vector4 {
public:
  Vector4();
  explicit Vector4(real_t _x, real_t _y, real_t _z, real_t _w);
  // access
  real_t operator[](int i) const;
  real_t &operator[](int i);
  Vector2 xy();
  Vector3 xyz();
  // arithmetic
  Vector4 operator+(const Vector4 &v) const;
  Vector4 &operator+=(const Vector4 &v);
  Vector4 operator-(const Vector4 &v) const;
  Vector4 &operator-=(const Vector4 &v);
  Vector4 operator*(real_t f) const;
  Vector4 &operator*=(real_t f);
  Vector4 operator/(real_t f) const;
  Vector4 &operator/=(real_t f);
  Vector4 operator-() const;
  // normalization
  real_t length2() const;
  real_t length() const;
  bool HasNaNs() const;

  friend std::ostream &operator<<(std::ostream &os, const Vector4 &v);
  real_t x, y, z, w;
};

typedef Vector2 vec2;
typedef Vector3 vec3;
typedef Vector4 vec4;

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
  real_t length2() const;
  real_t length() const;
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

typedef Vector<double, 3> vec3d;
typedef Vector<float, 2> vec2f;
typedef Vector<int, 2> ivec2;
typedef Vector<uint, 2> uivec2;
typedef Vector<int, 3> ivec3;
typedef Vector<uint, 3> uivec3;
typedef Vector<int, 4> ivec4;
typedef Vector<uint, 4> uivec4;

template <typename T, size_t D>
inline Vector<T, D> operator*(real_t f, const Vector<T, D> &v) {
  return v * f;
}

/* round
 * @v **[in]** vector
 * @return a vector with ceil applied to all components
 */
Vector<int, 3> ceil(const Vector3 &v);
/* round
 * @v **[in]** vector
 * @return a vector with floor applied to all components
 */
Vector<int, 3> floor(const Vector3 &v);

Vector<int, 3> min(Vector<int, 3> a, Vector<int, 3> b);
Vector<int, 3> max(Vector<int, 3> a, Vector<int, 3> b);

/* round
 * @v **[in]** vector
 * @return a vector with ceil applied to all components
 */
Vector<int, 2> ceil(const Vector2 &v);
/* round
 * @v **[in]** vector
 * @return a vector with floor applied to all components
 */
Vector<int, 2> floor(const Vector2 &v);

Vector<int, 2> min(Vector<int, 2> a, Vector<int, 2> b);
Vector<int, 2> max(Vector<int, 2> a, Vector<int, 2> b);

template <typename T, size_t D> T normalize(Vector<T, D> &v) {
  real_t d = v.length();
  for (size_t i = 0; i < D; i++)
    v[i] = v[i] / d;
}

} // namespace ponos

#endif
