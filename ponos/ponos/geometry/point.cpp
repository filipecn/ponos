#include <ponos/geometry/point.h>

#include <cmath>
#include <iostream>

namespace ponos {

Point2::Point2() { x = y = 0.f; }

Point2::Point2(real_t f) { x = y = f; }

Point2::Point2(const real_t *v) : x(v[0]), y(v[1]) { ASSERT(!HasNaNs()); }

Point2::Point2(real_t _x, real_t _y) : x(_x), y(_y) { ASSERT(!HasNaNs()); }

real_t Point2::operator[](int i) const {
  ASSERT(i >= 0 && i <= 1);
  return (&x)[i];
}

real_t &Point2::operator[](int i) {
  ASSERT(i >= 0 && i <= 1);
  return (&x)[i];
}

bool Point2::operator==(const Point2 &p) const {
  return IS_EQUAL(x, p.x) && IS_EQUAL(y, p.y);
}

Point2 Point2::operator+(const Vector2 &v) const {
  return Point2(x + v.x, y + v.y);
}

Point2 Point2::operator-(const Vector2 &v) const {
  return Point2(x - v.x, y - v.y);
}

Point2 Point2::operator-(const real_t &f) const { return Point2(x - f, y - f); }

Point2 Point2::operator+(const real_t &f) const { return Point2(x + f, y + f); }

Vector2 Point2::operator-(const Point2 &p) const {
  return Vector2(x - p.x, y - p.y);
};

Point2 Point2::operator/(real_t d) const { return Point2(x / d, y / d); }

Point2 Point2::operator*(real_t f) const { return Point2(x * f, y * f); }

Point2 &Point2::operator+=(const Vector2 &v) {
  x += v.x;
  y += v.y;
  return *this;
}

Point2 &Point2::operator-=(const Vector2 &v) {
  x -= v.x;
  y -= v.y;
  return *this;
}

Point2 &Point2::operator/=(real_t d) {
  x /= d;
  y /= d;
  return *this;
}

bool Point2::operator<(const Point2 &p) const {
  if (x >= p.x || y >= p.y)
    return false;
  return true;
}

bool Point2::operator>=(const Point2 &p) const { return x >= p.x && y >= p.y; }

bool Point2::operator<=(const Point2 &p) const { return x <= p.x && y <= p.y; }

bool Point2::HasNaNs() const { return std::isnan(x) || std::isnan(y); }

std::ostream &operator<<(std::ostream &os, const Point2 &p) {
  os << "[Point2] " << p.x << " " << p.y << std::endl;
  return os;
}

Point3::Point3() { x = y = z = 0.0f; }

Point3::Point3(real_t _x, real_t _y, real_t _z) : x(_x), y(_y), z(_z) {
  ASSERT(!HasNaNs());
}

Point3::Point3(const Vector3 &v) : x(v.x), y(v.y), z(v.z) {
  ASSERT(!HasNaNs());
}

Point3::Point3(const Point2 &p) : x(p.x), y(p.y), z(0.f) {}

Point3::Point3(const real_t *v) : x(v[0]), y(v[1]), z(v[2]) {
  ASSERT(!HasNaNs());
}

real_t Point3::operator[](int i) const {
  ASSERT(i >= 0 && i <= 2);
  return (&x)[i];
}

real_t &Point3::operator[](int i) {
  ASSERT(i >= 0 && i <= 2);
  return (&x)[i];
}

// arithmetic
Point3 Point3::operator+(const Vector3 &v) const {
  return Point3(x + v.x, y + v.y, z + v.z);
}

Point3 Point3::operator+(const real_t &f) const {
  return Point3(x + f, y + f, z + f);
}

Point3 Point3::operator-(const real_t &f) const {
  return Point3(x - f, y - f, z - f);
}

Point3 &Point3::operator+=(const Vector3 &v) {
  x += v.x;
  y += v.y;
  z += v.z;
  return *this;
}

Vector3 Point3::operator-(const Point3 &p) const {
  return Vector3(x - p.x, y - p.y, z - p.z);
}

Point3 Point3::operator-(const Vector3 &v) const {
  return Point3(x - v.x, y - v.y, z - v.z);
}

Point3 &Point3::operator-=(const Vector3 &v) {
  x -= v.x;
  y -= v.y;
  z -= v.z;
  return *this;
}

bool Point3::operator==(const Point3 &p) const {
  return IS_EQUAL(p.x, x) && IS_EQUAL(p.y, y) && IS_EQUAL(p.z, z);
}

bool Point3::operator>=(const Point3 &p) const {
  return x >= p.x && y >= p.y && z >= p.z;
}

bool Point3::operator<=(const Point3 &p) const {
  return x <= p.x && y <= p.y && z <= p.z;
}

Point3 Point3::operator*(real_t d) const { return Point3(x * d, y * d, z * d); }

Point3 Point3::operator/(real_t d) const { return Point3(x / d, y / d, z / d); }

Point3 &Point3::operator/=(real_t d) {
  x /= d;
  y /= d;
  z /= d;
  return *this;
}

bool Point3::operator==(const Point3 &p) {
  return IS_EQUAL(x, p.x) && IS_EQUAL(y, p.y) && IS_EQUAL(z, p.z);
}

Point2 Point3::xy() const { return Point2(x, y); }

Point2 Point3::yz() const { return Point2(y, z); }

Point2 Point3::xz() const { return Point2(x, z); }

Vector3 Point3::asVector3() const { return Vector3(x, y, z); }

ivec3 Point3::asIVec3() const {
  return ivec3(static_cast<const int &>(x), static_cast<const int &>(y),
               static_cast<const int &>(z));
}

bool Point3::HasNaNs() const {
  return std::isnan(x) || std::isnan(y) || std::isnan(z);
}

std::ostream &operator<<(std::ostream &os, const Point3 &p) {
  os << "[Point3] " << p.x << " " << p.y << " " << p.z << std::endl;
  return os;
}

///////////////////////////////////////////////////////////////////////////////

inline Point2 operator*(real_t f, const Point2 &p) { return p * f; }

inline real_t distance(const Point2 &a, const Point2 &b) {
  return (a - b).length();
}

inline real_t distance2(const Point2 &a, const Point2 &b) {
  return (a - b).length2();
}

inline real_t distance(const Point3 &a, const Point3 &b) {
  return (a - b).length();
}

inline real_t distance2(const Point3 &a, const Point3 &b) {
  return (a - b).length2();
}

} // namespace ponos
