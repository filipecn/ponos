#ifndef PONOS_GEOMETRY_POINT_H
#define PONOS_GEOMETRY_POINT_H

#include "geometry/utils.h"
#include "geometry/vector.h"
#include "log/debug.h"

namespace ponos {

class Point2 {
public:
  Point2();
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
  Vector2 operator-(const Point2 &p) const {
    return Vector2(x - p.x, y - p.y);
  };
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
    if (x >= p.x || y >= p.y)
      return false;
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

template <class T, int D> class Point {
public:
  Point();
  explicit Point(Point2 p);

  T operator[](int i) const {
    ASSERT(i >= 0 && i <= static_cast<int>(size));
    return v[i];
  }
  T &operator[](int i) {
    ASSERT(i >= 0 && i <= static_cast<int>(size));
    return v[i];
  }

  Point2 floatXY(size_t x = 0, size_t y = 1) {
    return Point2(static_cast<float>(v[x]), static_cast<float>(v[y]));
  }

  friend std::ostream &operator<<(std::ostream &os, const Point &p) {
    os << "[Point<" << D << ">]";
    for (int i = 0; i < p.size; i++)
      os << p[i] << " ";
    os << std::endl;
    return os;
  }

  size_t size;
  T v[D];
};

template <class T, int D> Point<T, D>::Point() {
  size = D;
  for (int i = 0; i < D; i++)
    v[i] = static_cast<T>(0);
}

template <class T, int D> Point<T, D>::Point(Point2 p) {
  size = D;
  v[0] = static_cast<T>(p.x);
  v[1] = static_cast<T>(p.y);
}

template <class T, int D>
inline bool operator==(const Point<T, D> &lhs, const Point<T, D> &rhs) {
  for (size_t i = 0; i < lhs.size; ++i)
    if (lhs[i] != rhs[i])
      return false;
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
  bool HasNaNs() const;

  friend std::ostream &operator<<(std::ostream &os, const Point3 &p);
  float x, y, z;
};

inline float distance(const Point3 &a, const Point3 &b) {
  return (a - b).length();
}

inline float distance2(const Point3 &a, const Point3 &b) {
  return (a - b).length2();
}

typedef Point3 point;

} // ponos namespace

#endif
