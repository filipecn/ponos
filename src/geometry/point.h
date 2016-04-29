#pragma once

#include "geometry/vector.h"
#include "log/debug.h"

namespace ponos {

  class Point3 {
  public:
    Point3();
    explicit Point3(float _x, float _y, float _z);
    // access
    float operator[](int i) const {
      ASSERT(i >= 0 && i <= 2);
      return (&x)[i];
    }
    float& operator[](int i) {
      ASSERT(i >= 0 && i <= 2);
      return (&x)[i];
    }
    // arithmetic
    Point3 operator+(const Vector3& v) const {
      return Point3(x + v.x, y + v.y, z + v.z);
    }
    Point3& operator+=(const Vector3& v) {
      x += v.x;
      y += v.y;
      z += v.z;
      return *this;
    }
    Vector3 operator-(const Point3& p) const {
      return Vector3(x - p.x, y - p.y, z - p.z);
    };
    Point3 operator-(const Vector3& v) const {
      return Point3(x - v.x, y - v.y, z - v.z);
    };
    Point3& operator-=(const Vector3& v) {
      x -= v.x;
      y -= v.y;
      z -= v.z;
      return *this;
    }
    Point3 operator/(float d) {
      return Point3(x / d, y / d, z / d);
    }
    Point3& operator/=(float d) {
      x /= d;
      y /= d;
      z /= d;
      return *this;
    }
    bool HasNaNs() const;

    float x, y, z;
  };

  inline float distance(const Point3& a, const Point3& b) {
    return (a - b).length();
  }

  inline float distance2(const Point3& a, const Point3& b) {
    return (a - b).length2();
  }

  typedef Point3 point;

}; // ponos namespace
