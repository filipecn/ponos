/// Copyright (c) 2017, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
///\file vector.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2017-08-19
///
///\brief

#ifndef PONOS_GEOMETRY_VECTOR_H
#define PONOS_GEOMETRY_VECTOR_H

#include <ponos/geometry/normal.h>
#include <ponos/log/debug.h>
#include <ponos/numeric/numeric.h>

#include <cstring>
#include <initializer_list>
#include <vector>

namespace ponos {

template<typename T> class Point2;
template<typename T> class Vector2 : public MathElement<T, 2u> {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, float>::value ||
                    std::is_same<T, double>::value,
                "Vector2 must hold an float type!");

public:
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  Vector2() = default;
  Vector2(T _x, T _y) : x(_x), y(_y) {}
  explicit Vector2(const Point2<T> &p) : x(p.x), y(p.y) {}
  explicit Vector2(const Normal2<T> &n) : x(n.x), y(n.y) {}
  explicit Vector2(T f) { x = y = f; }
  explicit Vector2(T *f) {
    x = f[0];
    y = f[1];
  }
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  // arithmetic
  Vector2 &operator+=(const Vector2 &v) {
    x += v.x;
    y += v.y;
    return *this;
  }
  Vector2 &operator-=(const Vector2 &v) {
    x -= v.x;
    y -= v.y;
    return *this;
  }
  Vector2 &operator*=(T f) {
    x *= f;
    y *= f;
    return *this;
  }
  Vector2 &operator/=(T f) {
    T inv = 1.f / f;
    x *= inv;
    y *= inv;
    return *this;
  }
  Vector2 operator-() const { return Vector2(-x, -y); }
  // ***********************************************************************
  //                         GETTERS & SETTERS
  // ***********************************************************************
  T operator[](size_t i) const { return (&x)[i]; }
  T &operator[](size_t i) { return (&x)[i]; }
  // ***********************************************************************
  //                              METHODS
  // ***********************************************************************
  T length2() const { return x * x + y * y; }
  T length() const { return sqrtf(length2()); }
  Vector2 right() const { return Vector2(y, -x); }
  Vector2 left() const { return Vector2(-y, x); }
  // ***********************************************************************
  //                           PUBLIC FIELDS
  // ***********************************************************************
  T x = T(0.0);
  T y = T(0.0);
};

// ***********************************************************************
//                           ARITHMETIC
// ***********************************************************************
template<typename T>
Vector2<T> operator+(const Vector2<T> &a, const Vector2<T> &b) {
  return Vector2<T>(a.x + b.x, a.y + b.y);
}
template<typename T>
Vector2<T> operator-(const Vector2<T> &a, const Vector2<T> &b) {
  return Vector2<T>(a.x - b.x, a.y - b.y);
}
template<typename T> Vector2<T> operator*(const Vector2<T> &a, T f) {
  return Vector2<T>(a.x * f, a.y * f);
}
template<typename T> Vector2<T> operator/(const Vector2<T> &a, T f) {
  T inv = 1.f / f;
  return Vector2<T>(a.x * inv, a.y * inv);
}
template<typename T> Vector2<T> operator*(T f, const Vector2<T> &v) {
  return v * f;
}
template<typename T> Vector2<T> operator/(T f, const Vector2<T> &v) {
  return Vector2<T>(f / v.x, f / v.y);
}
// ***********************************************************************
//                           BOOLEAN
// ***********************************************************************
template<typename T>
bool operator==(const Vector2<T> &a, const Vector2<T> &b) {
  return Check::is_equal(a.x, b.x) && Check::is_equal(a.y, b.y);
}
// ***********************************************************************
//                           ARITHMETIC
// ***********************************************************************
// GEOMETRY
template<typename T> T dot(const Vector2<T> &a, const Vector2<T> &b) {
  return a.x * b.x + a.y * b.y;
}
template<typename T> Vector2<T> normalize(const Vector2<T> &v) {
  return v / v.length();
}
template<typename T> Vector2<T> orthonormal(const Vector2<T> &v, bool first) {
  Vector2<T> n = normalize(v);
  if (first)
    return Vector2<T>(-n.y, n.x);
  return Vector2<T>(n.y, -n.x);
}
/** Projects a vector onto another.
 * \param a **[in]**
 * \param b **[in]**
 *
 * \returns the projection of **a** onto **b**
 */
template<typename T>
Vector2<T> project(const Vector2<T> &a, const Vector2<T> &b) {
  return (dot(b, a) / b.length2()) * b;
}
template<typename T> T cross(const Vector2<T> &a, const Vector2<T> &b) {
  return a.x * b.y - a.y * b.x;
}

template<typename T> class Point3;
template<typename T> class Vector3 : public MathElement<T, 3u> {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, float>::value ||
                    std::is_same<T, double>::value,
                "Vector3 must hold an float type!");

public:
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  Vector3() = default;
  explicit Vector3(T _f) : x(_f), y(_f), z(_f) {}
  Vector3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
  explicit Vector3(const T *v) {
    x = v[0];
    y = v[1];
    z = v[2];
  }
  explicit Vector3(const Normal3<T> &n) : x(n.x), y(n.y), z(n.z) {}
  explicit Vector3(const Point3<T> &p) : x(p.x), y(p.y), z(p.z) {}
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  Vector3 &operator=(const T &v) {
    x = y = z = v;
    return *this;
  }
  Vector3<T> &operator+=(const Vector3<T> &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }
  Vector3<T> &operator-=(const Vector3<T> &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }
  Vector3<T> &operator*=(T f) {
    x *= f;
    y *= f;
    z *= f;
    return *this;
  }
  Vector3<T> &operator/=(T f) {
    PONOS_CHECK_EXP(Check::is_zero(f))
    T inv = 1.f / f;
    x *= inv;
    y *= inv;
    z *= inv;
    return *this;
  }
  Vector3<T> &operator/=(const Vector3<T> &v) {
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
  }
  Vector3<T> operator-() const { return Vector3(-x, -y, -z); }
  // ***********************************************************************
  //                               ACCESS
  // ***********************************************************************
  T operator[](int i) const { return (&x)[i]; }
  T &operator[](int i) { return (&x)[i]; }
  Vector2<T> xy(int i = 0, int j = 1) const {
    return Vector2<T>((&x)[i], (&x)[j]);
  }
  // ***********************************************************************
  //                              METRICS
  // ***********************************************************************
  /// \note Also called L1-norm, taxicab norm, Manhattan norm
  /// \note Defined as ||v||_1 = sum_i(|v_i|)
  /// \return L1-norm of this vector
  T mLength() const { return std::abs(x) + std::abs(y) + std::abs(z); }
  /// \note Also called L2-norm, Euclidean norm, Euclidean distance, 2-norm
  /// \note Defined as ||v|| = (v_i * v_i)^(1/2)
  /// \return 2-norm of this vector
  T length() const { return std::sqrt(length2()); }
  /// \note Also called squared Euclidean distance
  /// \note Defined as ||v||^2 = v_i * v_i
  /// \return squared 2-norm of this vector
  T length2() const { return x * x + y * y + z * z; }
  /// \note Also called maximum norm, infinity norm
  /// \note Defined as ||v||_inf = argmax max(|v_i|)
  /// \return greatest absolute component value
  T maxAbs() const {
    if (std::abs(x) > std::abs(y) && std::abs(x) > std::abs(z))
      return x;
    if (std::abs(y) > std::abs(x) && std::abs(y) > std::abs(z))
      return y;
    return z;
  }
  /// \note Defined as argmax v_i
  /// \return greatest component value
  T max() const {
    if (x > y && x > z)
      return x;
    if (y > x && y > z)
      return y;
    return z;
  }
  /// \note Defined as argmax_i v_i
  /// \return Index of component with greatest value
  [[nodiscard]] int maxDimension() const {
    if (x > y && x > z)
      return 0;
    if (y > x && y > z)
      return 1;
    return 2;
  }
  /// \note Defined as argmax_i |v_i|
  /// \return Index of dimension with greatest value
  [[nodiscard]] int maxAbsDimension() const {
    if (std::abs(x) > std::abs(y) && std::abs(x) > std::abs(z))
      return 0;
    if (std::abs(y) > std::abs(x) && std::abs(y) > std::abs(z))
      return 1;
    return 2;
  }
  // ***********************************************************************
  //                             OPERATIONS
  // ***********************************************************************
  /// \note Normalization by vector length
  /// \note Defined as v / ||v||
  void normalize() {
    auto l = length();
    if (l != 0.f) {
      x /= l;
      y /= l;
      z /= l;
    }
  }
  /// \note Normalization by vector length
  /// \note Defined as v / ||v||
  /// \return Normalized vector of this vector
  Vector3 normalized() const {
    auto l = length();
    return (*this) / l;
  }
  /// \note b * dot(v,b) / ||b||
  /// \param b vector to project onto
  /// \return projection of this vector onto **b**
  Vector3 projectOnto(const Vector3 &b) {
    return (dot(b, *this) / b.length2()) * b;
  }
  /// \note v - b * dot(v,b) / ||b||
  /// \param b vector of rejection
  /// \return rejection of this vector on **b**
  Vector3 rejectOn(const Vector3 b) {
    return *this - (dot(b, *this) / b.length2()) * b;
  }
  // ***********************************************************************
  //                           PUBLIC FIELDS
  // ***********************************************************************
  T x = T(0.0);
  T y = T(0.0);
  T z = T(0.0);
};
// ARITHMETIC
template<typename T>
Vector3<T> operator+(const Vector3<T> &a, const Vector3<T> &b) {
  return Vector3<T>(a.x + b.x, a.y + b.y, a.z + b.z);
}
template<typename T>
Vector3<T> operator-(const Vector3<T> &a, const Vector3<T> &b) {
  return Vector3<T>(a.x - b.x, a.y - b.y, a.z - b.z);
}
template<typename T>
Vector3<T> operator*(const Vector3<T> &a, const Vector3<T> &b) {
  return Vector3<T>(a.x * b.x, a.y * b.y, a.z * b.z);
}
template<typename T> Vector3<T> operator*(const Vector3<T> &a, const T &f) {
  return Vector3<T>(a.x * f, a.y * f, a.z * f);
}
template<typename T> Vector3<T> operator/(const Vector3<T> &a, const T &f) {
  T inv = 1.f / f;
  return Vector3<T>(a.x * inv, a.y * inv, a.z * inv);
}
template<typename T> Vector3<T> operator*(T f, const Vector3<T> &v) {
  return v * f;
}
// BOOLEAN
template<typename T>
bool operator==(const Vector3<T> &a, const Vector3<T> &b) {
  return Check::is_equal(a.x, b.x) && Check::is_equal(a.y, b.y) &&
      Check::is_equal(a.z, b.z);
}
template<typename T> bool operator<(const Vector3<T> &a, const Vector3<T> &b) {
  if (a.x < b.x)
    return true;
  if (a.y < b.y)
    return true;
  return a.z < b.z;
}
template<typename T> bool operator>(const Vector3<T> &a, const Vector3<T> &b) {
  if (a.x > b.x)
    return true;
  if (a.y > b.y)
    return true;
  return a.z > b.z;
}
template<typename T>
bool operator>=(const Vector3<T> &a, const Vector3<T> &b) {
  return a.x >= b.x && a.y >= b.y && a.z >= b.z;
}
template<typename T>
bool operator<=(const Vector3<T> &a, const Vector3<T> &b) {
  return a.x <= b.x && a.y <= b.y && a.z <= b.z;
}
// GEOMETRY
template<typename T> T dot(const Vector3<T> &a, const Vector3<T> &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
template<typename T>
Vector3<T> cross(const Vector3<T> &a, const Vector3<T> &b) {
  return Vector3<T>((a.y * b.z) - (a.z * b.y), (a.z * b.x) - (a.x * b.z),
                    (a.x * b.y) - (a.y * b.x));
}
template<typename T>
T triple(const Vector3<T> &a, const Vector3<T> &b, const Vector3<T> &c) {
  return dot(a, cross(b, c));
}
template<typename T> Vector3<T> normalize(const Vector3<T> &v) {
  if (v.length2() == 0.f)
    return v;
  return v / v.length();
}
/** \brief compute the two orthogonal-tangential vectors from a
 * \param a **[in]** normal
 * \param b **[out]** first tangent
 * \param c **[out]** second tangent
 */
template<typename T>
void tangential(const Vector3<T> &a, Vector3<T> &b, Vector3<T> &c) {
  b = normalize(cross(a, ((std::abs(a.y) > 0.f || std::abs(a.z) > 0.f)
                          ? Vector3<T>(1, 0, 0)
                          : Vector3<T>(0, 1, 1))));
  c = normalize(cross(a, b));
}
/// \note b * dot(a,b) / ||b||
/// \tparam T
/// \param a
/// \param b
/// \return projection of **a** onto **b**
template<typename T>
Vector3<T> project(const Vector3<T> &a, const Vector3<T> &b) {
  return (dot(b, a) / b.length2()) * b;
}
template<typename T> Vector3<T> cos(const Vector3<T> &v) {
  return Vector3<T>(std::cos(v.x), std::cos(v.y), std::cos(v.z));
}
template<typename T> Vector3<T> max(const Vector3<T> &a, const Vector3<T> &b) {
  return Vector3<T>(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}
template<typename T> Vector3<T> abs(const Vector3<T> &a) {
  return Vector3<T>(std::abs(a.x), std::abs(a.y), std::abs(a.z));
}

template<typename T> class Vector4 : public MathElement<T, 4> {
public:
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  Vector4() = default;
  explicit Vector4(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) {}
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  Vector4<T> &operator+=(const Vector4<T> &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;
    return *this;
  }
  Vector4<T> &operator-=(const Vector4<T> &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;
    return *this;
  }
  Vector4<T> &operator*=(T f) {
    x *= f;
    y *= f;
    z *= f;
    w *= f;
    return *this;
  }
  Vector4<T> &operator/=(T f) {
    T inv = 1.f / f;
    x *= inv;
    y *= inv;
    z *= inv;
    w *= inv;
    return *this;
  }
  Vector4<T> operator-() const { return Vector4(-x, -y, -z, -w); }
  // ***********************************************************************
  //                         GETTERS & SETTERS
  // ***********************************************************************
  T operator[](int i) const { return (&x)[i]; }
  T &operator[](int i) { return (&x)[i]; }
  Vector2<T> xy() { return Vector2<T>(x, y); }
  Vector3<T> xyz() { return Vector3<T>(x, y, z); }
  // ***********************************************************************
  //                              METHODS
  // ***********************************************************************
  T length2() const { return x * x + y * y + z * z + w * w; }
  T length() const { return sqrtf(length2()); }
  // ***********************************************************************
  //                           PUBLIC FIELDS
  // ***********************************************************************
  T x = T(0.0);
  T y = T(0.0);
  T z = T(0.0);
  T w = T(0.0);
};
// ARITHMETIC
template<typename T>
Vector4<T> operator+(const Vector4<T> &a, const Vector4<T> &b) {
  return Vector4<T>(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
template<typename T>
Vector4<T> operator-(const Vector4<T> &a, const Vector4<T> &b) {
  return Vector4<T>(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
template<typename T> Vector4<T> operator*(const Vector4<T> &a, T f) {
  return Vector4<T>(a.x * f, a.y * f, a.z * f, a.w * f);
}
template<typename T> Vector4<T> operator/(const Vector4<T> &a, T f) {
  T inv = 1.f / f;
  return Vector4<T>(a.x * inv, a.y * inv, a.z * inv, a.w * inv);
}

using vec2 = Vector2<real_t>;
using vec3 = Vector3<real_t>;
using vec4 = Vector4<real_t>;
using vec3d = Vector3<double>;
using vec3f = Vector3<float>;
using vec2f = Vector2<float>;

template<typename T>
std::ostream &operator<<(std::ostream &os, const Vector2<T> &v) {
  os << "Vector2 [" << v.x << " " << v.y << "]";
  return os;
}
template<typename T>
std::ostream &operator<<(std::ostream &os, const Vector3<T> &v) {
  os << "Vector3 [" << v.x << " " << v.y << " " << v.z << "]";
  return os;
}
template<typename T>
std::ostream &operator<<(std::ostream &os, const Vector4<T> &v) {
  os << "Vector4 [" << v.x << " " << v.y << " " << v.z << " " << v.w << "]";
  return os;
}

} // namespace ponos

// std hash support
namespace std {

template<typename T> struct hash<ponos::Vector2<T>> {
  size_t operator()(ponos::Vector2<T> const &v) const {
    hash<T> hasher;
    size_t s = 0;
    // inject x component
    size_t h = hasher(v.x);
    h += 0x9e3779b9 + (s << 6) + (s >> 2);
    s ^= h;
    // inject y component
    h = hasher(v.y);
    h += 0x9e3779b9 + (s << 6) + (s >> 2);
    s ^= h;
    return s;
  }
};

template<typename T> struct hash<ponos::Vector3<T>> {
  size_t operator()(ponos::Vector3<T> const &v) const {
    hash<T> hasher;
    size_t s = 0;
    // inject x component
    size_t h = hasher(v.x);
    h += 0x9e3779b9 + (s << 6) + (s >> 2);
    s ^= h;
    // inject y component
    h = hasher(v.y);
    h += 0x9e3779b9 + (s << 6) + (s >> 2);
    s ^= h;
    // inject y component
    h = hasher(v.z);
    h += 0x9e3779b9 + (s << 6) + (s >> 2);
    s ^= h;
    return s;
  }
};

} // namespace std

#endif
