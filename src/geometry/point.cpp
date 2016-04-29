#include "geometry/point.h"

namespace ponos {

Point3::Point3() { x = y = z = 0.0f; }

Point3::Point3(float _x, float _y, float _z)
  : x(_x), y(_y), z(_z) {
  ASSERT(!HasNaNs());
}

bool Point3::HasNaNs() const {
  return isnan(x) || isnan(y) || isnan(z);
}

}; // ponos namespace
