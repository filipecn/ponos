#include "geometry/vector.h"

#include <cmath>

namespace ponos {

Vector2::Vector2() { x = y = 0.0f; }

Vector2::Vector2(float _x, float _y)
  : x(_x), y(_y) {
  ASSERT(!HasNaNs());
}

bool Vector2::HasNaNs() const {
  return std::isnan(x) || std::isnan(y);
}

Vector3::Vector3() { x = y = z = 0.0f; }

Vector3::Vector3(float _x, float _y, float _z)
  : x(_x), y(_y), z(_z) {
  ASSERT(!HasNaNs());
}

Vector3::Vector3(const Normal& n)
  : x(n.x), y(n.y), z(n.z) {}

bool Vector3::HasNaNs() const {
  return std::isnan(x) || std::isnan(y) || std::isnan(z);
}

} // ponos namespace
