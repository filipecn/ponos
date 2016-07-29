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

Vector4::Vector4() { x = y = z = w = 0.0f; }

Vector4::Vector4(float _x, float _y, float _z, float _w)
  : x(_x), y(_y), z(_z), w(_w) {
  ASSERT(!HasNaNs());
}

bool Vector4::HasNaNs() const {
  return std::isnan(x) || std::isnan(y) || std::isnan(z) || std::isnan(w);
}


} // ponos namespace
