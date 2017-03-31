#include "geometry/point.h"

#include <iostream>
#include <cmath>

namespace ponos {

Point2::Point2() { x = y = 0.f; }

Point2::Point2(float f) { x = y = f; }

Point2::Point2(float _x, float _y) : x(_x), y(_y) { ASSERT(!HasNaNs()); }

bool Point2::HasNaNs() const { return std::isnan(x) || std::isnan(y); }

std::ostream &operator<<(std::ostream &os, const Point2 &p) {
  os << "[Point2] " << p.x << " " << p.y << std::endl;
  return os;
}

Point3::Point3() { x = y = z = 0.0f; }

Point3::Point3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {
  ASSERT(!HasNaNs());
}

Point3::Point3(const Vector3 &v) : x(v.x), y(v.y), z(v.z) {
  ASSERT(!HasNaNs());
}

Point3::Point3(const float *v) : x(v[0]), y(v[1]), z(v[2]) {
  ASSERT(!HasNaNs());
}

bool Point3::HasNaNs() const {
  return std::isnan(x) || std::isnan(y) || std::isnan(z);
}

std::ostream &operator<<(std::ostream &os, const Point3 &p) {
  os << "[Point3] " << p.x << " " << p.y << " " << p.z << std::endl;
  return os;
}

} // ponos namespace
