#include <ponos/geometry/point.h>
#include <ponos/geometry/vector.h>

#include <cmath>

namespace ponos {

Vector2::Vector2() { x = y = 0.0f; }

Vector2::Vector2(float _x, float _y) : x(_x), y(_y) { ASSERT(!HasNaNs()); }

Vector3::Vector3(const float *v) {
  x = v[0];
  y = v[1];
  z = v[2];
}

Vector2::Vector2(const Point2 &p) : x(p.x), y(p.y) {}

Vector2::Vector2(const Normal2D &n) : x(n.x), y(n.y) {}

Vector2::Vector2(float f) { x = y = f; }

Vector2::Vector2(float *f) {
  x = f[0];
  y = f[1];
}

real_t Vector2::operator[](int i) const {
  ASSERT(i >= 0 && i <= 1);
  return (&x)[i];
}

real_t &Vector2::operator[](int i) {
  ASSERT(i >= 0 && i <= 1);
  return (&x)[i];
}

// arithmetic
Vector2 Vector2::operator+(const Vector2 &v) const {
  return Vector2(x + v.x, y + v.y);
}

Vector2 &Vector2::operator+=(const Vector2 &v) {
  x += v.x;
  y += v.y;
  return *this;
}

Vector2 Vector2::operator-(const Vector2 &v) const {
  return Vector2(x - v.x, y - v.y);
}

Vector2 &Vector2::operator-=(const Vector2 &v) {
  x -= v.x;
  y -= v.y;
  return *this;
}

Vector2 Vector2::operator*(real_t f) const { return Vector2(x * f, y * f); }

Vector2 &Vector2::operator*=(real_t f) {
  x *= f;
  y *= f;
  return *this;
}

Vector2 Vector2::operator/(real_t f) const {
  CHECK_FLOAT_EQUAL(f, 0.f);
  real_t inv = 1.f / f;
  return Vector2(x * inv, y * inv);
}

Vector2 &Vector2::operator/=(real_t f) {
  CHECK_FLOAT_EQUAL(f, 0.f);
  real_t inv = 1.f / f;
  x *= inv;
  y *= inv;
  return *this;
}

Vector2 Vector2::operator-() const { return Vector2(-x, -y); }

bool Vector2::operator==(const Vector2 &v) {
  return IS_EQUAL(x, v.x) && IS_EQUAL(y, v.y);
}

real_t Vector2::length2() const { return x * x + y * y; }

real_t Vector2::length() const { return sqrtf(length2()); }

Vector2 Vector2::right() const { return Vector2(y, -x); }

Vector2 Vector2::left() const { return Vector2(-y, x); }

bool Vector2::HasNaNs() const { return std::isnan(x) || std::isnan(y); }

std::ostream &operator<<(std::ostream &os, const Vector2 &v) {
  os << "[vector2]" << v.x << " " << v.y << std::endl;
  return os;
}

Vector3::Vector3() { x = y = z = 0.0f; }

Vector3::Vector3(float _f) : x(_f), y(_f), z(_f) { ASSERT(!HasNaNs()); }

Vector3::Vector3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {
  ASSERT(!HasNaNs());
}

Vector3::Vector3(const Normal &n) : x(n.x), y(n.y), z(n.z) {}

Vector3::Vector3(const Point3 &p) : x(p.x), y(p.y), z(p.z) {}

bool Vector3::operator==(const Vector3 &v) {
  return IS_EQUAL(x, v.x) && IS_EQUAL(y, v.y) && IS_EQUAL(z, v.z);
}

bool Vector3::operator<(const Vector3 &v) {
  if (x < v.x)
    return true;
  if (y < v.y)
    return true;
  if (z < v.z)
    return true;
  return false;
}

bool Vector3::operator>(const Vector3 &v) {
  if (x > v.x)
    return true;
  if (y > v.y)
    return true;
  if (z > v.z)
    return true;
  return false;
}

real_t Vector3::operator[](int i) const {
  ASSERT(i >= 0 && i <= 2);
  return (&x)[i];
}

real_t &Vector3::operator[](int i) {
  ASSERT(i >= 0 && i <= 2);
  return (&x)[i];
}

Vector2 Vector3::xy() { return Vector2(x, y); }

Vector3 &Vector3::operator=(const real_t &v) {
  x = y = z = v;
  return *this;
}

Vector3 Vector3::operator+(const Vector3 &v) const {
  return Vector3(x + v.x, y + v.y, z + v.z);
}

Vector3 &Vector3::operator+=(const Vector3 &v) {
  x += v.x;
  y += v.y;
  z += v.z;
  return *this;
}

Vector3 Vector3::operator-(const Vector3 &v) const {
  return Vector3(x - v.x, y - v.y, z - v.z);
}

Vector3 &Vector3::operator-=(const Vector3 &v) {
  x -= v.x;
  y -= v.y;
  z -= v.z;
  return *this;
}

Vector3 Vector3::operator*(const Vector3 &v) const {
  return Vector3(x * v.x, y * v.y, z * v.z);
}

Vector3 Vector3::operator*(real_t f) const {
  return Vector3(x * f, y * f, z * f);
}

Vector3 &Vector3::operator*=(real_t f) {
  x *= f;
  y *= f;
  z *= f;
  return *this;
}

Vector3 Vector3::operator/(real_t f) const {
  CHECK_FLOAT_EQUAL(f, 0.f);
  real_t inv = 1.f / f;
  return Vector3(x * inv, y * inv, z * inv);
}

Vector3 &Vector3::operator/=(real_t f) {
  CHECK_FLOAT_EQUAL(f, 0.f);
  real_t inv = 1.f / f;
  x *= inv;
  y *= inv;
  z *= inv;
  return *this;
}

Vector3 &Vector3::operator/=(const Vector3 &v) {
  x /= v.x;
  y /= v.y;
  z /= v.z;
  return *this;
}

Vector3 Vector3::operator-() const { return Vector3(-x, -y, -z); }

bool Vector3::operator>=(const Vector3 &p) const {
  return x >= p.x && y >= p.y && z >= p.z;
}

bool Vector3::operator<=(const Vector3 &p) const {
  return x <= p.x && y <= p.y && z <= p.z;
}

real_t Vector3::length2() const { return x * x + y * y + z * z; }

real_t Vector3::length() const { return sqrtf(length2()); }

bool Vector3::HasNaNs() const {
  return std::isnan(x) || std::isnan(y) || std::isnan(z);
}

std::ostream &operator<<(std::ostream &os, const Vector3 &v) {
  os << "[vector3]" << v.x << " " << v.y << " " << v.z << std::endl;
  return os;
}

Vector4::Vector4() { x = y = z = w = 0.0f; }

Vector4::Vector4(float _x, float _y, float _z, float _w)
    : x(_x), y(_y), z(_z), w(_w) {
  ASSERT(!HasNaNs());
}

real_t Vector4::operator[](int i) const {
  ASSERT(i >= 0 && i <= 3);
  return (&x)[i];
}

real_t &Vector4::operator[](int i) {
  ASSERT(i >= 0 && i <= 3);
  return (&x)[i];
}

Vector2 Vector4::xy() { return Vector2(x, y); }

Vector3 Vector4::xyz() { return Vector3(x, y, z); }

// arithmetic
Vector4 Vector4::operator+(const Vector4 &v) const {
  return Vector4(x + v.x, y + v.y, z + v.z, w + v.w);
}

Vector4 &Vector4::operator+=(const Vector4 &v) {
  x += v.x;
  y += v.y;
  z += v.z;
  w += v.w;
  return *this;
}

Vector4 Vector4::operator-(const Vector4 &v) const {
  return Vector4(x - v.x, y - v.y, z - v.z, w - v.w);
}

Vector4 &Vector4::operator-=(const Vector4 &v) {
  x -= v.x;
  y -= v.y;
  z -= v.z;
  w -= v.w;
  return *this;
}

Vector4 Vector4::operator*(real_t f) const {
  return Vector4(x * f, y * f, z * f, w * f);
}

Vector4 &Vector4::operator*=(real_t f) {
  x *= f;
  y *= f;
  z *= f;
  w *= f;
  return *this;
}

Vector4 Vector4::operator/(real_t f) const {
  CHECK_FLOAT_EQUAL(f, 0.f);
  real_t inv = 1.f / f;
  return Vector4(x * inv, y * inv, z * inv, w * inv);
}

Vector4 &Vector4::operator/=(real_t f) {
  CHECK_FLOAT_EQUAL(f, 0.f);
  real_t inv = 1.f / f;
  x *= inv;
  y *= inv;
  z *= inv;
  w *= inv;
  return *this;
}

Vector4 Vector4::operator-() const { return Vector4(-x, -y, -z, -w); }

real_t Vector4::length2() const {
  return x * x + y * y + z * z + w * w;
  ;
}

real_t Vector4::length() const { return sqrtf(length2()); }

bool Vector4::HasNaNs() const {
  return std::isnan(x) || std::isnan(y) || std::isnan(z) || std::isnan(w);
}

std::ostream &operator<<(std::ostream &os, const Vector4 &v) {
  os << "[vector4]" << v.x << " " << v.y << " " << v.z << " " << v.w
     << std::endl;
  return os;
}

///////////////////////////////////////////////////////////////////////////////
Vector2 operator*(real_t f, const Vector2 &v) { return v * f; }
Vector2 operator/(real_t f, const Vector2 &v) {
  return Vector2(f / v.x, f / v.y);
}

real_t dot(const Vector2 &a, const Vector2 &b) { return a.x * b.x + a.y * b.y; }

Vector2 normalize(const Vector2 &v) { return v / v.length(); }

Vector2 orthonormal(const Vector2 &v, bool first) {
  Vector2 n = normalize(v);
  if (first)
    return Vector2(-n.y, n.x);
  return Vector2(n.y, -n.x);
}

Vector2 project(const Vector2 &a, const Vector2 &b) {
  return (dot(b, a) / b.length2()) * b;
}

real_t cross(const Vector2 &a, const Vector2 &b) {
  return a.x * b.y - a.y * b.x;
}

Vector3 operator*(real_t f, const Vector3 &v) { return v * f; }

real_t dot(const Vector3 &a, const Vector3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vector3 cross(const Vector3 &a, const Vector3 &b) {
  return Vector3((a.y * b.z) - (a.z * b.y), (a.z * b.x) - (a.x * b.z),
                 (a.x * b.y) - (a.y * b.x));
}

real_t triple(const Vector3 &a, const Vector3 &b, const Vector3 &c) {
  return dot(a, cross(b, c));
}

Vector3 normalize(const Vector3 &v) {
  if (v.length2() == 0.f)
    return v;
  return v / v.length();
}

void tangential(const Vector3 &a, Vector3 &b, Vector3 &c) {
  b = normalize(cross(a, ((std::fabs(a.y) > 0.f || std::fabs(a.z) > 0.f)
                              ? Vector3(1, 0, 0)
                              : Vector3(0, 1, 1))));
  c = normalize(cross(a, b));
}

Vector3 cos(const Vector3 &v) {
  return Vector3(cosf(v.x), cosf(v.y), cosf(v.z));
}

Vector3 max(const Vector3 &a, const Vector3 &b) {
  return Vector3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

Vector<int, 3> ceil(const Vector3 &v) {
  return Vector<int, 3>(static_cast<int>(v[0] + 0.5f),
                        static_cast<int>(v[1] + 0.5f),
                        static_cast<int>(v[2] + 0.5f));
}

Vector<int, 3> floor(const Vector3 &v) {
  return Vector<int, 3>(static_cast<int>(v[0]), static_cast<int>(v[1]),
                        static_cast<int>(v[2]));
}

Vector<int, 3> min(Vector<int, 3> a, Vector<int, 3> b) {
  return Vector<int, 3>(std::min(a[0], b[0]), std::min(a[1], b[1]),
                        std::min(a[2], b[2]));
}

Vector<int, 3> max(Vector<int, 3> a, Vector<int, 3> b) {
  return Vector<int, 3>(std::max(a[0], b[0]), std::max(a[1], b[1]),
                        std::max(a[2], b[2]));
}

Vector<int, 2> ceil(const Vector2 &v) {
  return Vector<int, 2>(static_cast<int>(v[0] + 0.5f),
                        static_cast<int>(v[1] + 0.5f));
}

Vector<int, 2> floor(const Vector2 &v) {
  return Vector<int, 2>(static_cast<int>(v[0]), static_cast<int>(v[1]));
}

Vector<int, 2> min(Vector<int, 2> a, Vector<int, 2> b) {
  return Vector<int, 2>(std::min(a[0], b[0]), std::min(a[1], b[1]));
}

Vector<int, 2> max(Vector<int, 2> a, Vector<int, 2> b) {
  return Vector<int, 2>(std::max(a[0], b[0]), std::max(a[1], b[1]));
}

} // namespace ponos
