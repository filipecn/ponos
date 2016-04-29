#pragma once

#include "geometry/normal.h"
#include "log/debug.h"

namespace ponos {

  class Normal;

  class Vector2 {
  public:
    Vector2();
    explicit Vector2(float _x, float _y);
    // access
    float operator[](int i) const {
      ASSERT(i >= 0 && i <= 1);
      return (&x)[i];
    }
    float& operator[](int i) {
      ASSERT(i >= 0 && i <= 1);
      return (&x)[i];
    }
    // arithmetic
    Vector2 operator+(const Vector2& v) const {
      return Vector2(x + v.x, y + v.y);
    }
    Vector2& operator+=(const Vector2& v) {
      x += v.x;
      y += v.y;
      return *this;
    }
    Vector2 operator-(const Vector2& v) const {
      return Vector2(x - v.x, y - v.y);
    }
    Vector2& operator-=(const Vector2& v) {
      x -= v.x;
      y -= v.y;
      return *this;
    }
    Vector2 operator*(float f) const {
      return Vector2(x * f, y * f);
    }
    Vector2& operator*=(float f) {
      x *= f;
      y *= f;
      return *this;
    }
    Vector2 operator/(float f) const {
      CHECK_FLOAT_EQUAL(f, 0.f);
      float inv = 1.f / f;
      return Vector2(x * inv, y * inv);
    }
    Vector2& operator/=(float f) {
      CHECK_FLOAT_EQUAL(f, 0.f);
      float inv = 1.f / f;
      x *= inv;
      y *= inv;
      return *this;
    }
    Vector2 operator-() const {
      return Vector2(-x, -y);
    }
    // normalization
    float length2() const {
      return x * x + y * y;
    }
    float length() const {
      return sqrtf(length2());
    }
    bool HasNaNs() const;

    float x, y;
  };

  inline Vector2 operator*(float f, const Vector2& v) {
    return v*f;
  }

  inline float dot(const Vector2& a, const Vector2& b) {
    return a.x * b.x + a.y * b.y;
  }

  inline Vector2 normalize(const Vector2& v) {
    return v / v.length();
  }

  class Vector3 {
  public:
    Vector3();
    explicit Vector3(float _x, float _y, float _z);
    explicit Vector3(const Normal& n);
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
    Vector3 operator+(const Vector3& v) const {
      return Vector3(x + v.x, y + v.y, z + v.z);
    }
    Vector3& operator+=(const Vector3& v) {
      x += v.x;
      y += v.y;
      z += v.z;
      return *this;
    }
    Vector3 operator-(const Vector3& v) const {
      return Vector3(x - v.x, y - v.y, z - v.z);
    }
    Vector3& operator-=(const Vector3& v) {
      x -= v.x;
      y -= v.y;
      z -= v.z;
      return *this;
    }
    Vector3 operator*(float f) const {
      return Vector3(x * f, y * f, z * f);
    }
    Vector3& operator*=(float f) {
      x *= f;
      y *= f;
      z *= f;
      return *this;
    }
    Vector3 operator/(float f) const {
      CHECK_FLOAT_EQUAL(f, 0.f);
      float inv = 1.f / f;
      return Vector3(x * inv, y * inv, z * inv);
    }
    Vector3& operator/=(float f) {
      CHECK_FLOAT_EQUAL(f, 0.f);
      float inv = 1.f / f;
      x *= inv;
      y *= inv;
      z *= inv;
      return *this;
    }
    Vector3 operator -() const {
      return Vector3(-x, -y, -z);
    }
    // normalization
    float length2() const {
      return x * x + y * y + z * z;
    }
    float length() const {
      return sqrtf(length2());
    }
    bool HasNaNs() const;

    float x, y, z;
  };

  inline Vector3 operator*(float f, const Vector3& v) {
    return v*f;
  }

  inline float dot(const Vector3& a, const Vector3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  inline Vector3 cross(const Vector3& a, const Vector3& b) {
    return Vector3((a.y * b.z) - (a.z * b.y),
                   (a.z * b.x) - (a.x * b.z),
                   (a.x * b.y) - (a.y * b.x));
  }

  inline Vector3 normalize(const Vector3& v) {
    return v / v.length();
  }

  typedef Vector2 vec2;
  typedef Vector3 vec3;

}; // ponos namespace
