#include "geometry/vector.h"

#include "geometry/point.h"

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

std::ostream& operator<<(std::ostream& os, const Vector2& v) {
	os << "[vector2]" << v.x << " " << v.y << std::endl;
	return os;
}

Vector3::Vector3() { x = y = z = 0.0f; }

Vector3::Vector3(float _f)
  : x(_f), y(_f), z(_f) {
  ASSERT(!HasNaNs());
}

Vector3::Vector3(float _x, float _y, float _z)
  : x(_x), y(_y), z(_z) {
  ASSERT(!HasNaNs());
}

Vector3::Vector3(const Normal& n)
  : x(n.x), y(n.y), z(n.z) {}

Vector3::Vector3(const Point3& p)
  : x(p.x), y(p.y), z(p.z) {}

bool Vector3::HasNaNs() const {
  return std::isnan(x) || std::isnan(y) || std::isnan(z);
}

std::ostream& operator<<(std::ostream& os, const Vector3& v) {
	os << "[vector3]" << v.x << " " << v.y << " " << v.z << std::endl;
	return os;
}


Vector4::Vector4() { x = y = z = w = 0.0f; }

Vector4::Vector4(float _x, float _y, float _z, float _w)
  : x(_x), y(_y), z(_z), w(_w) {
  ASSERT(!HasNaNs());
}

bool Vector4::HasNaNs() const {
  return std::isnan(x) || std::isnan(y) || std::isnan(z) || std::isnan(w);
}

std::ostream& operator<<(std::ostream& os, const Vector4& v) {
	os << "[vector4]" << v.x << " " << v.y << " " << v.z << " " << v.w << std::endl;
	return os;
}

} // ponos namespace
