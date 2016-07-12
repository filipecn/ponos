#include "geometry/point.h"

#include <cmath>

namespace ponos {

  Point2::Point2() { x = y = 0.f; }

  Point2::Point2(float _x, float _y)
  : x(_x), y(_y) {
    ASSERT(!HasNaNs());
  }

  bool Point2::HasNaNs() const {
    return std::isnan(x) || std::isnan(y);
  }

  Point3::Point3() { x = y = z = 0.0f; }

  Point3::Point3(float _x, float _y, float _z)
  : x(_x), y(_y), z(_z) {
    ASSERT(!HasNaNs());
  }

  bool Point3::HasNaNs() const {
    return std::isnan(x) || std::isnan(y) || std::isnan(z);
  }

} // ponos namespace
