#pragma once

#include "geometry/vector.h"

#include <iostream>

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
    Normal& operator*=(float f) {
      x *= f;
      y *= f;
      z *= f;
      return *this;
    }
		friend std::ostream& operator<<(std::ostream& os, const Normal& n) {
			os << "[Normal] " << n.x << " " << n.y << " " << n.z << std::endl;
			return os;
		}
    float x, y, z;
  };

//  inline Normal faceForward(const Normal& n, const Vector3& v) {
//    return (dot(n, v) < 0.f) ? -n : n;
//  }

} // ponos namespace
