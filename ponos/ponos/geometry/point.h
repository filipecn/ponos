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

#ifndef PONOS_GEOMETRY_POINT_H
#define PONOS_GEOMETRY_POINT_H

#include <initializer_list>

#include <ponos/geometry/utils.h>
#include <ponos/geometry/vector.h>
#include <ponos/log/debug.h>

namespace ponos {

template <typename T> class Point2 {
public:
  typedef T ScalarType;
  Point2();
  Point2(real_t f);
  explicit Point2(const real_t *v);
  explicit Point2(real_t _x, real_t _y);

  // access
  T operator[](int i) const;
  T &operator[](int i);
  bool operator==(const Point2 &p) const;
  Point2 operator+(const Vector2<T> &v) const;
  Point2 operator-(const Vector2<T> &v) const;
  Point2 operator-(const real_t &f) const;
  Point2 operator+(const real_t &f) const;
  Vector2<T> operator-(const Point2 &p) const;
  Point2 operator/(real_t d) const;
  Point2 operator*(real_t f) const;
  Point2 &operator+=(const Vector2<T> &v);
  Point2 &operator-=(const Vector2<T> &v);
  Point2 &operator/=(real_t d);
  bool operator<(const Point2 &p) const;
  bool operator>=(const Point2 &p) const;
  bool operator<=(const Point2 &p) const;
  bool HasNaNs() const;
  template <typename TT>
  friend std::ostream &operator<<(std::ostream &os, const Point2<TT> &p);
  T x, y;
};

typedef Point2<real_t> point2;
typedef Point2<uint> point2u;
typedef Point2<int> point2i;
typedef Point2<float> point2f;
typedef Point2<double> point2d;

template <typename T> Point2<T> operator*(real_t f, const Point2<T> &p);
template <typename T> real_t distance(const Point2<T> &a, const Point2<T> &b);
template <typename T> real_t distance2(const Point2<T> &a, const Point2<T> &b);

template <typename T> class Point3 {
public:
  Point3();
  explicit Point3(real_t _x, real_t _y, real_t _z);
  explicit Point3(const Vector3<T> &v);
  explicit Point3(const real_t *v);
  explicit Point3(real_t v);
  explicit Point3(const Point2<T> &p);
  explicit operator Vector3<T>() const { return Vector3<T>(x, y, z); }
  // access
  T operator[](int i) const;
  T &operator[](int i);
  // arithmetic
  Point3 operator+(const Vector3<T> &v) const;
  Point3 operator+(const Point3<T> &v) const;
  Point3 operator+(const real_t &f) const;
  Point3 operator-(const real_t &f) const;
  Point3 &operator+=(const Vector3<T> &v);
  Vector3<T> operator-(const Point3 &p) const;
  Point3 operator-(const Vector3<T> &v) const;
  Point3 &operator-=(const Vector3<T> &v);
  bool operator==(const Point3 &p) const;
  bool operator>=(const Point3 &p) const;
  bool operator<=(const Point3 &p) const;
  Point3 operator*(real_t d) const;
  Point3 operator/(real_t d) const;
  Point3 &operator/=(real_t d);
  Point3 &operator*=(T d);
  // boolean
  bool operator==(const Point3 &p);
  Point2<T> xy() const;
  Point2<T> yz() const;
  Point2<T> xz() const;
  Vector3<T> asVector3() const;
  ivec3 asIVec3() const;
  bool HasNaNs() const;
  static uint dimension() { return 3; }
  template <typename TT>
  friend std::ostream &operator<<(std::ostream &os, const Point3<TT> &p);
  T x, y, z;
};

typedef Point3<real_t> point3;
typedef Point3<uint> point3u;
typedef Point3<int> point3i;
typedef Point3<float> point3f;
typedef Point3<float> point3d;

template <typename T> real_t distance(const Point3<T> &a, const Point3<T> &b);
template <typename T> real_t distance2(const Point3<T> &a, const Point3<T> &b);

template <class T, size_t D> class Point {
public:
  Point();
  Point(T v);
  explicit Point(Point2<T> p);
  Point(std::initializer_list<T> p);
  T operator[](int i) const;
  T &operator[](int i);
  bool operator>=(const Point<T, D> &p) const;
  bool operator<=(const Point<T, D> &p) const;
  Vector<T, D> operator-(const Point<T, D> &p) const;
  Point<T, D> operator+(const Vector<T, D> &V) const;
  Point2<T> floatXY(size_t x = 0, size_t y = 1) const;
  friend std::ostream &operator<<(std::ostream &os, const Point &p) {
    os << "[Point<" << D << ">]";
    for (size_t i = 0; i < p.size; i++)
      os << p[i] << " ";
    os << std::endl;
    return os;
  }

  size_t size;
  T v[D];
};

#include "point.inl"

} // namespace ponos

#endif
