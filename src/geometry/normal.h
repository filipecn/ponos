#pragma once

#include "geometry/vector.h"

namespace ponos {

  class Vector3;

  class Normal {
  public:
    explicit Normal(float _x, float _y, float _z);
    explicit Normal(const Vector3& v);
    Normal() { x = y = z = 0.; }

    Normal operator-() const {
      return Normal(-x, -y, -z);
    }

    float x, y, z;
  };

//  inline Normal faceForward(const Normal& n, const Vector3& v) {
//    return (dot(n, v) < 0.f) ? -n : n;
//  }

}; // ponos namespace
