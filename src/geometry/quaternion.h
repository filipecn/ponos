#pragma once

#include "geometry/transform.h"
#include "geometry/vector.h"

namespace ponos {

  class Quaternion {
  public:
    Quaternion();
    Quaternion(Vector3 _v, float _w);
    Quaternion operator+(const Quaternion& q) const {
      return Quaternion(v + q.v, w + q.w);
    }
    Quaternion& operator+=(const Quaternion& q) {
      v += q.v;
      w += q.w;
      return *this;
    }
    Quaternion operator-(const Quaternion& q) const {
      return Quaternion(v - q.v, w - q.w);
    }
    Quaternion& operator-=(const Quaternion& q) {
      v -= q.v;
      w -= q.w;
      return *this;
    }
    Quaternion operator/(float d) const {
      return Quaternion(v / d, w / d);
    }

    Transform toTransform() const {
      float m[4][4];
      m[0][0] = 1.f - 2.f * (v.x * v.x + v.z * v.z);
      m[0][1] = 2.f * (v.x * v.y + v.z * w);
      m[0][2] = 2.f * v.x * v.z - v.y * w;
      m[0][3] = 0.f;

      m[1][0] = 2.f * (v.x * v.y - v.z * w);
      m[1][1] = 1.f - 2.f * (v.x * v.x + v.z * v.z);
      m[1][2] = 2.f * (v.y * v.z + v.x * w);
      m[1][3] = 0.f;

      m[2][0] = 2.f * (v.x * v.z + v.y * w);
      m[2][1] = 2.f * (v.y * v.z - v.x * w);
      m[2][2] = 1.f - 2.f * (v.x * v.x + v.y * v.y);
      m[2][3] = 0.f;

      m[3][0] = 0.f;
      m[3][1] = 0.f;
      m[3][2] = 0.f;
      m[3][3] = 1.f;

      return Transform(m);
    }

    Vector3 v;
    float w;
  };

  inline float dot(const Quaternion& q1, const Quaternion& q2) {
    return dot(q1.v, q2.v) + q1.w * q2.w;
  }

  inline Quaternion normalize(const Quaternion& q) {
    return q / sqrtf(dot(q, q));
  }

} // ponos namespace
