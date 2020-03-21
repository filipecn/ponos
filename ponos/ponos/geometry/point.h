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
///\file point.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2019-08-18
///
///\brief

#ifndef PONOS_GEOMETRY_POINT_H
#define PONOS_GEOMETRY_POINT_H

#include <initializer_list>

#include <ponos/geometry/utils.h>
#include <ponos/geometry/vector.h>
#include <ponos/log/debug.h>

namespace ponos {

///\brief
///
///\tparam T
template <typename T> class Point2 {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, float>::value ||
                    std::is_same<T, double>::value,
                "Point2 must hold a float type!");

public:
  typedef T ScalarType;
  Point2(T f = T(0)) { x = y = f; }
  explicit Point2(const real_t *v) : x(v[0]), y(v[1]) {}
  explicit Point2(real_t _x, real_t _y) : x(_x), y(_y) {}
  // access
  T operator[](int i) const { return (&x)[i]; }
  T &operator[](int i) { return (&x)[i]; }
  // arithmetic
  Point2 &operator+=(const Vector2<T> &v) {
    x += v.x;
    y += v.y;
    return *this;
  }
  Point2 &operator-=(const Vector2<T> &v) {
    x -= v.x;
    y -= v.y;
    return *this;
  }
  Point2 &operator/=(real_t d) {
    x /= d;
    y /= d;
    return *this;
  }
  T x = T(0.0);
  T y = T(0.0);
};

// ARITHMETIC
template <typename T>
Point2<T> operator+(const Point2<T> a, const Vector2<T> &v) {
  return Point2<T>(a.x + v.x, a.y + v.y);
}
template <typename T>
Point2<T> operator-(const Point2<T> a, const Vector2<T> &v) {
  return Point2<T>(a.x - v.x, a.y - v.y);
}
template <typename T> Point2<T> operator+(const Point2<T> a, const T &f) {
  return Point2<T>(a.x + f, a.y + f);
}
template <typename T> Point2<T> operator-(const Point2<T> a, const T &f) {
  return Point2<T>(a.x - f, a.y - f);
}
template <typename T>
Vector2<T> operator-(const Point2<T> &a, const Point2<T> &b) {
  return Vector2<T>(a.x - b.x, a.y - b.y);
}
template <typename T> Point2<T> operator/(const Point2<T> a, real_t f) {
  return Point2<T>(a.x / f, a.y / f);
}
template <typename T> Point2<T> operator*(const Point2<T> a, real_t f) {
  return Point2<T>(a.x * f, a.y * f);
}
template <typename T> Point2<T> operator*(real_t f, const Point2<T> &a) {
  return Point2<T>(a.x * f, a.y * f);
}
// BOOLEAN
template <typename T> bool operator==(const Point2<T> &a, const Point2<T> &b) {
  return Check::is_equal(a.x, b.x) && Check::is_equal(a.y, b.y);
}
template <typename T> bool operator<(const Point2<T> &a, const Point2<T> &b) {
  return !(a.x >= b.x || a.y >= b.y);
}
template <typename T> bool operator>=(const Point2<T> &a, const Point2<T> &b) {
  return a.x >= b.x && a.y >= b.y;
}
template <typename T> bool operator<=(const Point2<T> &a, const Point2<T> &b) {
  return a.x <= b.x && a.y <= b.y;
}
// GEOMETRY
template <typename T> real_t distance(const Point2<T> &a, const Point2<T> &b) {
  return (a - b).length();
}
template <typename T> real_t distance2(const Point2<T> &a, const Point2<T> &b) {
  return (a - b).length2();
}

template <typename T> class Point3 {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, float>::value ||
                    std::is_same<T, double>::value,
                "Size2 must hold an float type!");

public:
  Point3() {}
  explicit Point3(real_t v) { x = y = z = v; }
  explicit Point3(real_t _x, real_t _y, real_t _z) : x(_x), y(_y), z(_z) {}
  explicit Point3(const Vector3<T> &v) : x(v.x), y(v.y), z(v.z) {}
  explicit Point3(const Point2<T> &p) : x(p.x), y(p.y), z(0) {}
  explicit Point3(const real_t *v) : x(v[0]), y(v[1]), z(v[2]) {}
  explicit operator Vector3<T>() const { return Vector3<T>(x, y, z); }
  // access
  T operator[](int i) const { return (&x)[i]; }
  T &operator[](int i) { return (&x)[i]; }
  Point2<T> xy() const { return Point2<T>(x, y); }
  Point2<T> yz() const { return Point2<T>(y, z); }
  Point2<T> xz() const { return Point2<T>(x, z); }
  // arithmetic
  Point3 &operator+=(const Vector3<T> &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }
  Point3 &operator-=(const Vector3<T> &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }
  Point3 &operator/=(real_t d) {
    x /= d;
    y /= d;
    z /= d;
    return *this;
  }
  Point3 &operator*=(T d) {
    x *= d;
    y *= d;
    z *= d;
    return *this;
  }
  Vector3<T> asVector3() const { return Vector3<T>(x, y, z); }
  static uint dimension() { return 3; }
  T x = T(0.0);
  T y = T(0.0);
  T z = T(0.0);
};

// ARITHMETIC
template <typename T>
Vector3<T> operator-(const Point3<T> &a, const Point3<T> &b) {
  return Vector3<T>(a.x - b.x, a.y - b.y, a.z - b.z);
}
template <typename T>
Point3<T> operator-(const Point3<T> &a, const Vector3<T> &v) {
  return Point3<T>(a.x - v.x, a.y - v.y, a.z - v.z);
}
template <typename T>
Point3<T> operator+(const Point3<T> &a, const Vector3<T> &v) {
  return Point3<T>(a.x + v.x, a.y + v.y, a.z + v.z);
}
template <typename T> Point3<T> operator+(const Point3<T> &a, const real_t &f) {
  return Point3<T>(a.x + f, a.y + f, a.z + f);
}
template <typename T> Point3<T> operator-(const Point3<T> &a, const real_t &f) {
  return Point3<T>(a.x - f, a.y - f, a.z - f);
}
template <typename T> Point3<T> operator*(const Point3<T> &a, real_t f) {
  return Point3<T>(a.x * f, a.y * f, a.z * f);
}
template <typename T> Point3<T> operator/(const Point3<T> &a, real_t f) {
  return Point3<T>(a.x / f, a.y / f, a.z / f);
}
// BOOLEAN
template <typename T> bool operator==(const Point3<T> &a, const Point3<T> &b) {
  return Check::is_equal(a.x, b.x) && Check::is_equal(a.y, b.y) &&
         Check::is_equal(a.z, b.z);
}
template <typename T> bool operator!=(const Point3<T> &a, const Point3<T> &b) {
  return !Check::is_equal(a.x, b.x) || !Check::is_equal(a.y, b.y) ||
         !Check::is_equal(a.z, b.z);
}
template <typename T> bool operator>=(const Point3<T> &a, const Point3<T> &b) {
  return a.x >= b.x && a.y >= b.y && a.z >= b.z;
}
template <typename T> bool operator<=(const Point3<T> &a, const Point3<T> &b) {
  return a.x <= b.x && a.y <= b.y && a.z <= b.z;
}
// GEOMETRY
template <typename T> real_t distance(const Point3<T> &a, const Point3<T> &b) {
  return (a - b).length();
}
template <typename T> real_t distance2(const Point3<T> &a, const Point3<T> &b) {
  return (a - b).length2();
}

using point2 = Point2<real_t>;
using point2f = Point2<float>;
using point2d = Point2<double>;

using point3 = Point3<real_t>;
using point3f = Point3<float>;
using point3d = Point3<double>;

template <typename T>
std::ostream &operator<<(std::ostream &os, const Point2<T> &p) {
  os << "Point2[" << p.x << " " << p.y << "]";
  return os;
}
template <typename T>
std::ostream &operator<<(std::ostream &os, const Point3<T> &p) {
  os << "Point3[" << p.x << " " << p.y << " " << p.z << "]";
  return os;
}

} // namespace ponos

// std hash support
namespace std {

template <typename T> struct hash<ponos::Point2<T>> {
  size_t operator()(ponos::Point2<T> const &v) const {
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

template <typename T> struct hash<ponos::Point3<T>> {
  size_t operator()(ponos::Point3<T> const &v) const {
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
