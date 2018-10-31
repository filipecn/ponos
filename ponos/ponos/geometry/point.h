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

class Point2 {
 public:
  typedef float ScalarType;
  Point2();
  Point2(float f);
  explicit Point2(const float *v);
  explicit Point2(float _x, float _y);

  // access
  float operator[](int i) const {
    ASSERT(i >= 0 && i <= 1);
    return (&x)[i];
  }
  float &operator[](int i) {
    ASSERT(i >= 0 && i <= 1);
    return (&x)[i];
  }

  bool operator==(const Point2 &p) const {
    return IS_EQUAL(x, p.x) && IS_EQUAL(y, p.y);
  }
  Point2 operator+(const Vector2 &v) const { return Point2(x + v.x, y + v.y); }
  Point2 operator-(const Vector2 &v) const { return Point2(x - v.x, y - v.y); }
  Point2 operator-(const float &f) const { return Point2(x - f, y - f); }
  Point2 operator+(const float &f) const { return Point2(x + f, y + f); }
  Vector2 operator-(const Point2 &p) const {
    return Vector2(x - p.x, y - p.y);
  };
  Point2 operator/(float d) const { return Point2(x / d, y / d); }
  Point2 operator*(float f) const { return Point2(x * f, y * f); }
  Point2 &operator+=(const Vector2 &v) {
    x += v.x;
    y += v.y;
    return *this;
  }
  Point2 &operator-=(const Vector2 &v) {
    x -= v.x;
    y -= v.y;
    return *this;
  }
  Point2 &operator/=(float d) {
    x /= d;
    y /= d;
    return *this;
  }

  bool operator<(const Point2 &p) const {
    if (x >= p.x || y >= p.y) return false;
    return true;
  }

  bool operator>=(const Point2 &p) const { return x >= p.x && y >= p.y; }

  bool operator<=(const Point2 &p) const { return x <= p.x && y <= p.y; }

  bool HasNaNs() const;
  friend std::ostream &operator<<(std::ostream &os, const Point2 &p);
  float x, y;
};

inline Point2 operator*(float f, const Point2 &p) { return p * f; }

inline float distance(const Point2 &a, const Point2 &b) {
  return (a - b).length();
}

inline float distance2(const Point2 &a, const Point2 &b) {
  return (a - b).length2();
}

template <class T, int D>
class Point {
 public:
  Point();
  Point(T v);
  explicit Point(Point2 p);
  Point(std::initializer_list<T> p) {
    size = D;
    int k = 0;
    for (auto it = p.begin(); it != p.end(); ++it) {
      if (k >= D) break;
      v[k++] = *it;
    }
  }

  T operator[](int i) const {
    ASSERT(i >= 0 && i <= static_cast<int>(size));
    return v[i];
  }
  T &operator[](int i) {
    ASSERT(i >= 0 && i <= static_cast<int>(size));
    return v[i];
  }
  bool operator>=(const Point<T, D> &p) const {
    for (int i = 0; i < D; i++)
      if (v[i] < p[i]) return false;
    return true;
  }
  bool operator<=(const Point<T, D> &p) const {
    for (int i = 0; i < D; i++)
      if (v[i] > p[i]) return false;
    return true;
  }

  Vector<T, D> operator-(const Point<T, D> &p) const {
    Vector<T, D> V;
    for (int i = 0; i < D; i++) V[i] = v[i] - p[i];
    return V;
  }

  Point<T, D> operator+(const Vector<T, D> &V) const {
    Point<T, D> P;
    for (int i = 0; i < D; i++) P[i] = v[i] + V[i];
    return P;
  }

  Point2 floatXY(size_t x = 0, size_t y = 1) const {
    return Point2(static_cast<float>(v[x]), static_cast<float>(v[y]));
  }

  friend std::ostream &operator<<(std::ostream &os, const Point &p) {
    os << "[Point<" << D << ">]";
    for (size_t i = 0; i < p.size; i++) os << p[i] << " ";
    os << std::endl;
    return os;
  }

  size_t size;
  T v[D];
};

template <class T, int D>
Point<T, D>::Point() {
  size = D;
  for (int i = 0; i < D; i++) v[i] = static_cast<T>(0);
}

template <class T, int D>
Point<T, D>::Point(T v) {
  size = D;
  for (int i = 0; i < D; i++) v[i] = static_cast<T>(v);
}

template <class T, int D>
Point<T, D>::Point(Point2 p) {
  size = D;
  v[0] = static_cast<T>(p.x);
  v[1] = static_cast<T>(p.y);
}

template <class T, int D>
inline bool operator==(const Point<T, D> &lhs, const Point<T, D> &rhs) {
  for (size_t i = 0; i < lhs.size; ++i)
    if (lhs[i] != rhs[i]) return false;
  return true;
}

template <class T, int D>
inline bool operator!=(const Point<T, D> &lhs, const Point<T, D> &rhs) {
  return !(lhs == rhs);
}

class Point3 {
 public:
  Point3();
  explicit Point3(float _x, float _y, float _z);
  explicit Point3(const Vector3 &v);
  explicit Point3(const float *v);
  explicit Point3(const Point2 &p);
  explicit operator Vector3() const { return Vector3(x, y, z); }
  // access
  float operator[](int i) const {
    ASSERT(i >= 0 && i <= 2);
    return (&x)[i];
  }
  float &operator[](int i) {
    ASSERT(i >= 0 && i <= 2);
    return (&x)[i];
  }
  // arithmetic
  Point3 operator+(const Vector3 &v) const {
    return Point3(x + v.x, y + v.y, z + v.z);
  }
  Point3 operator+(const float &f) const { return Point3(x + f, y + f, z + f); }
  Point3 operator-(const float &f) const { return Point3(x - f, y - f, z - f); }
  Point3 &operator+=(const Vector3 &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }
  Vector3 operator-(const Point3 &p) const {
    return Vector3(x - p.x, y - p.y, z - p.z);
  };
  Point3 operator-(const Vector3 &v) const {
    return Point3(x - v.x, y - v.y, z - v.z);
  };
  Point3 &operator-=(const Vector3 &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }
  bool operator==(const Point3 &p) const {
    return IS_EQUAL(p.x, x) && IS_EQUAL(p.y, y) && IS_EQUAL(p.z, z);
  }

  bool operator>=(const Point3 &p) const {
    return x >= p.x && y >= p.y && z >= p.z;
  }

  bool operator<=(const Point3 &p) const {
    return x <= p.x && y <= p.y && z <= p.z;
  }
  Point3 operator*(float d) const { return Point3(x * d, y * d, z * d); }
  Point3 operator/(float d) const { return Point3(x / d, y / d, z / d); }
  Point3 &operator/=(float d) {
    x /= d;
    y /= d;
    z /= d;
    return *this;
  }
  // boolean
  bool operator==(const Point3 &p) {
    return IS_EQUAL(x, p.x) && IS_EQUAL(y, p.y) && IS_EQUAL(z, p.z);
  }

  Point2 xy() const { return Point2(x, y); }
  Point2 yz() const { return Point2(y, z); }
  Point2 xz() const { return Point2(x, z); }
  Vector3 asVector3() const { return Vector3(x, y, z); }
  ivec3 asIVec3() const {
    return ivec3(static_cast<const int &>(x), static_cast<const int &>(y),
                 static_cast<const int &>(z));
  }
  bool HasNaNs() const;
  static uint dimension() { return 3; }
  friend std::ostream &operator<<(std::ostream &os, const Point3 &p);
  float x, y, z;
};

inline float distance(const Point3 &a, const Point3 &b) {
  return (a - b).length();
}

inline float distance2(const Point3 &a, const Point3 &b) {
  return (a - b).length2();
}

typedef Point<int, 2> Point2i;
typedef Point<float, 2> Point2f;
}  // namespace ponos

#endif
