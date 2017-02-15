#include "geometry/vector.h"

#include "geometry/point.h"

#include <cmath>

namespace ponos {

Vector2::Vector2() { x = y = 0.0f; }

Vector2::Vector2(float _x, float _y)
  : x(_x), y(_y) {
  ASSERT(!HasNaNs());
}

Vector2::Vector2(const Point2& p)
  : x(p.x), y(p.y) {}

Vector2::Vector2(float f) { x = y = f; }

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

Vector<3, int> ceil(const Vector3 &v) {
	return Vector<3, int>(
			static_cast<int>(v[0] + 0.5f),
			static_cast<int>(v[1] + 0.5f),
			static_cast<int>(v[2] + 0.5f));
}

Vector<3, int> floor(const Vector3 &v) {
	return Vector<3, int>(
			static_cast<int>(v[0]),
			static_cast<int>(v[1]),
			static_cast<int>(v[2]));
}

Vector<3, int> min(Vector<3, int> a, Vector<3, int> b) {
	return Vector<3, int>(
			std::min(a[0], b[0]),
			std::min(a[1], b[1]),
			std::min(a[2], b[2]));
}

Vector<3, int> max(Vector<3, int> a, Vector<3, int> b) {
	return Vector<3, int>(
			std::max(a[0], b[0]),
			std::max(a[1], b[1]),
			std::max(a[2], b[2]));
}


Vector<2, int> ceil(const Vector2 &v) {
	return Vector<2, int>(
			static_cast<int>(v[0] + 0.5f),
			static_cast<int>(v[1] + 0.5f));
}

Vector<2, int> floor(const Vector2 &v) {
	return Vector<2, int>(
			static_cast<int>(v[0]),
			static_cast<int>(v[1]));
}

Vector<2, int> min(Vector<2, int> a, Vector<2, int> b) {
	return Vector<2, int>(
			std::min(a[0], b[0]),
			std::min(a[1], b[1]));
}

Vector<2, int> max(Vector<2, int> a, Vector<2, int> b) {
	return Vector<2, int>(
			std::max(a[0], b[0]),
			std::max(a[1], b[1]));
}


} // ponos namespace
