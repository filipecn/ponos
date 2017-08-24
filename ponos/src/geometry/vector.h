/*
 * Copyright (c) 2017 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
*/

#ifndef PONOS_GEOMETRY_VECTOR_H
#define PONOS_GEOMETRY_VECTOR_H

#include "geometry/normal.h"
#include "geometry/numeric.h"
#include "log/debug.h"

#include <cstring>
#include <initializer_list>
#include <vector>

namespace ponos {
template <typename T> class DenseVector {
public:
  DenseVector() {}
  DenseVector(size_t n) { set(n); }
  DenseVector(const DenseVector<T> &V) { v = V.v; }
  DenseVector<T> &operator=(DenseVector<T> other) {
    v = other.v;
    return *this;
  }
  void set(size_t n) {
    v.resize(n);
    std::fill(v.begin(), v.end(), 0);
  }
  T operator[](size_t i) const { return v[i]; }
  T &operator[](size_t i) { return v[i]; }
  friend std::ostream &operator<<(std::ostream &os, const DenseVector<T> &m) {
    for (size_t j = 0; j < m.size(); j++)
      os << m[j] << " ";
    os << std::endl;
    return os;
  }
  template <typename... Args> void emplace_back(Args &&... args) {
    v.emplace_back(std::forward<Args>(args)...);
  }
  void resize(size_t n) { v.resize(n); }

  size_t size() const { return v.size(); }

private:
  std::vector<T> v;
};

class Point2;
class Vector2 {
public:
  Vector2();
  explicit Vector2(float _x, float _y);
  explicit Vector2(const Point2 &p);
  explicit Vector2(const Normal2D &n);
  Vector2(float f);
  Vector2(float *f);
  // access
  float operator[](int i) const {
    ASSERT(i >= 0 && i <= 1);
    return (&x)[i];
  }
  float &operator[](int i) {
    ASSERT(i >= 0 && i <= 1);
    return (&x)[i];
  }
  // arithmetic
  Vector2 operator+(const Vector2 &v) const {
    return Vector2(x + v.x, y + v.y);
  }
  Vector2 &operator+=(const Vector2 &v) {
    x += v.x;
    y += v.y;
    return *this;
  }
  Vector2 operator-(const Vector2 &v) const {
    return Vector2(x - v.x, y - v.y);
  }
  Vector2 &operator-=(const Vector2 &v) {
    x -= v.x;
    y -= v.y;
    return *this;
  }
  Vector2 operator*(float f) const { return Vector2(x * f, y * f); }
  Vector2 &operator*=(float f) {
    x *= f;
    y *= f;
    return *this;
  }
  Vector2 operator/(float f) const {
    CHECK_FLOAT_EQUAL(f, 0.f);
    float inv = 1.f / f;
    return Vector2(x * inv, y * inv);
  }
  Vector2 &operator/=(float f) {
    CHECK_FLOAT_EQUAL(f, 0.f);
    float inv = 1.f / f;
    x *= inv;
    y *= inv;
    return *this;
  }
  Vector2 operator-() const { return Vector2(-x, -y); }
  bool operator==(const Vector2 &v) {
    return IS_EQUAL(x, v.x) && IS_EQUAL(y, v.y);
  }
  // normalization
  float length2() const { return x * x + y * y; }
  float length() const { return sqrtf(length2()); }
  Vector2 right() const { return Vector2(y, -x); }
  Vector2 left() const { return Vector2(-y, x); }
  bool HasNaNs() const;

  friend std::ostream &operator<<(std::ostream &os, const Vector2 &v);
  float x, y;
};

inline Vector2 operator*(float f, const Vector2 &v) { return v * f; }
inline Vector2 operator/(float f, const Vector2 &v) {
  return Vector2(f / v.x, f / v.y);
}

inline float dot(const Vector2 &a, const Vector2 &b) {
  return a.x * b.x + a.y * b.y;
}

template <typename T>
inline T dot(const std::vector<T> &a, const std::vector<T> &b) {
  ASSERT_FATAL(a.size() == b.size());
  T sum = T(0);
  for (size_t i = 0; i < a.size(); i++)
    sum = sum + a[i] * b[i];
  return sum;
}

inline Vector2 normalize(const Vector2 &v) { return v / v.length(); }

inline Vector2 orthonormal(const Vector2 &v, bool first = true) {
  Vector2 n = normalize(v);
  if (first)
    return Vector2(-n.y, n.x);
  return Vector2(n.y, -n.x);
}

inline float cross(const Vector2 &a, const Vector2 &b) {
  return a.x * b.y - a.y * b.x;
}

/** Projects a vector onto another.
 * \param a **[in]**
 * \param b **[in]**
 *
 * \returns the projection of **a** onto **b**
 */
inline Vector2 project(const Vector2 &a, const Vector2 &b) {
  return (dot(b, a) / b.length2()) * b;
}

class Point3;
class Vector3 {
public:
  Vector3();
  explicit Vector3(float _f);
  explicit Vector3(float _x, float _y, float _z);
  explicit Vector3(const Normal &n);
  explicit Vector3(const Point3 &p);
  // boolean
  bool operator==(const Vector3 &v) {
    return IS_EQUAL(x, v.x) && IS_EQUAL(y, v.y) && IS_EQUAL(z, v.z);
  }
  bool operator<(const Vector3 &v) {
    if (x < v.x)
      return true;
    if (y < v.y)
      return true;
    if (z < v.z)
      return true;
    return false;
  }
  bool operator>(const Vector3 &v) {
    if (x > v.x)
      return true;
    if (y > v.y)
      return true;
    if (z > v.z)
      return true;
    return false;
  }
  // access
  float operator[](int i) const {
    ASSERT(i >= 0 && i <= 2);
    return (&x)[i];
  }
  float &operator[](int i) {
    ASSERT(i >= 0 && i <= 2);
    return (&x)[i];
  }
  Vector2 xy() { return Vector2(x, y); }
  // arithmetic
  Vector3 &operator=(const float &v) {
    x = y = z = v;
    return *this;
  }
  Vector3 operator+(const Vector3 &v) const {
    return Vector3(x + v.x, y + v.y, z + v.z);
  }
  Vector3 &operator+=(const Vector3 &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }
  Vector3 operator-(const Vector3 &v) const {
    return Vector3(x - v.x, y - v.y, z - v.z);
  }
  Vector3 &operator-=(const Vector3 &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }
  Vector3 operator*(const Vector3 &v) const {
    return Vector3(x * v.x, y * v.y, z * v.z);
  }
  Vector3 operator*(float f) const { return Vector3(x * f, y * f, z * f); }
  Vector3 &operator*=(float f) {
    x *= f;
    y *= f;
    z *= f;
    return *this;
  }
  Vector3 operator/(float f) const {
    CHECK_FLOAT_EQUAL(f, 0.f);
    float inv = 1.f / f;
    return Vector3(x * inv, y * inv, z * inv);
  }
  Vector3 &operator/=(float f) {
    CHECK_FLOAT_EQUAL(f, 0.f);
    float inv = 1.f / f;
    x *= inv;
    y *= inv;
    z *= inv;
    return *this;
  }
  Vector3 &operator/=(const Vector3 &v) {
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
  }
  Vector3 operator-() const { return Vector3(-x, -y, -z); }
  // normalization
  float length2() const { return x * x + y * y + z * z; }
  float length() const { return sqrtf(length2()); }
  bool HasNaNs() const;

  friend std::ostream &operator<<(std::ostream &os, const Vector3 &v);
  float x, y, z;
};

inline Vector3 operator*(float f, const Vector3 &v) { return v * f; }

inline float dot(const Vector3 &a, const Vector3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vector3 cross(const Vector3 &a, const Vector3 &b) {
  return Vector3((a.y * b.z) - (a.z * b.y), (a.z * b.x) - (a.x * b.z),
                 (a.x * b.y) - (a.y * b.x));
}

inline Vector3 normalize(const Vector3 &v) {
  if (v.length2() == 0.f)
    return v;
  return v / v.length();
}
/** \brief compute the two orthogonal-tangential vectors from a
 * \param a **[in]** normal
 * \param b **[out]** first tangent
 * \param c **[out]** second tangent
 */
inline void tangential(const Vector3 &a, Vector3 &b, Vector3 &c) {
  b = normalize(cross(a, ((std::fabs(a.y) > 0.f || std::fabs(a.z) > 0.f)
                              ? Vector3(1, 0, 0)
                              : Vector3(0, 1, 1))));
  c = normalize(cross(a, b));
}

inline Vector3 cos(const Vector3 &v) {
  return Vector3(cosf(v.x), cosf(v.y), cosf(v.z));
}

inline Vector3 max(const Vector3 &a, const Vector3 &b) {
  return Vector3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

class Vector4 {
public:
  Vector4();
  explicit Vector4(float _x, float _y, float _z, float _w);
  // access
  float operator[](int i) const {
    ASSERT(i >= 0 && i <= 3);
    return (&x)[i];
  }
  float &operator[](int i) {
    ASSERT(i >= 0 && i <= 3);
    return (&x)[i];
  }
  Vector2 xy() { return Vector2(x, y); }
  Vector3 xyz() { return Vector3(x, y, z); }
  // arithmetic
  Vector4 operator+(const Vector4 &v) const {
    return Vector4(x + v.x, y + v.y, z + v.z, w + v.w);
  }
  Vector4 &operator+=(const Vector4 &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;
    return *this;
  }
  Vector4 operator-(const Vector4 &v) const {
    return Vector4(x - v.x, y - v.y, z - v.z, w - v.w);
  }
  Vector4 &operator-=(const Vector4 &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;
    return *this;
  }
  Vector4 operator*(float f) const {
    return Vector4(x * f, y * f, z * f, w * f);
  }
  Vector4 &operator*=(float f) {
    x *= f;
    y *= f;
    z *= f;
    w *= f;
    return *this;
  }
  Vector4 operator/(float f) const {
    CHECK_FLOAT_EQUAL(f, 0.f);
    float inv = 1.f / f;
    return Vector4(x * inv, y * inv, z * inv, w * inv);
  }
  Vector4 &operator/=(float f) {
    CHECK_FLOAT_EQUAL(f, 0.f);
    float inv = 1.f / f;
    x *= inv;
    y *= inv;
    z *= inv;
    w *= inv;
    return *this;
  }
  Vector4 operator-() const { return Vector4(-x, -y, -z, -w); }
  // normalization
  float length2() const {
    return x * x + y * y + z * z + w * w;
    ;
  }
  float length() const { return sqrtf(length2()); }
  bool HasNaNs() const;

  friend std::ostream &operator<<(std::ostream &os, const Vector4 &v);
  float x, y, z, w;
};

typedef Vector2 vec2;
typedef Vector3 vec3;
typedef Vector4 vec4;

template <typename T = float, size_t D = 3> class Vector {
public:
  Vector() {
    size = D;
    memset(v, 0, D * sizeof(T));
  }
  Vector(std::initializer_list<T> values) : Vector() {
    int k = 0;
    for (auto value = values.begin(); value != values.end(); value++)
      v[k++] = *value;
  }
  Vector(size_t n, const T *t) : Vector() {
    for (size_t i = 0; i < D && i < n; i++)
      v[i] = t[i];
  }

  Vector(const T &t) : Vector() {
    for (size_t i = 0; i < D; i++)
      v[i] = t;
  }

  Vector(const T &x, const T &y) : Vector() {
    if (size > 1) {
      v[0] = x;
      v[1] = y;
    }
  }

  Vector(const T &x, const T &y, const T &z) : Vector() {
    if (size > 2) {
      v[0] = x;
      v[1] = y;
      v[2] = z;
    }
  }

  T operator[](int i) const {
    ASSERT(i >= 0 && i <= static_cast<int>(size));
    return v[i];
  }
  T &operator[](int i) {
    ASSERT(i >= 0 && i <= static_cast<int>(size));
    return v[i];
  }
  bool operator==(const Vector<T, D> &_v) const {
    for (size_t i = 0; i < size; i++)
      if (!IS_EQUAL(v[i], _v[i]))
        return false;
    return true;
  }
  bool operator!=(const Vector<T, D> &_v) const {
    bool dif = false;
    for (size_t i = 0; i < size; i++)
      if (!IS_EQUAL(v[i], _v[i])) {
        dif = true;
        break;
      }
    return dif;
  }
  bool operator<=(const Vector<T, D> &_v) const {
    for (size_t i = 0; i < size; i++)
      if (v[i] > _v[i])
        return false;
    return true;
  }
  bool operator<(const Vector<T, D> &_v) const {
    for (size_t i = 0; i < size; i++)
      if (v[i] >= _v[i])
        return false;
    return true;
  }
  bool operator>=(const Vector<T, D> &_v) const {
    for (size_t i = 0; i < size; i++)
      if (v[i] < _v[i])
        return false;
    return true;
  }
  bool operator>(const Vector<T, D> &_v) const {
    for (size_t i = 0; i < size; i++)
      if (v[i] <= _v[i])
        return false;
    return true;
  }

  Vector<T, D> operator-(const Vector<T, D> &_v) const {
    Vector<T, D> v_;
    for (size_t i = 0; i < D; i++)
      v_[i] = v[i] - _v[i];
    return v_;
  }

  Vector<T, D> operator+(const Vector<T, D> &_v) const {
    Vector<T, D> v_;
    for (size_t i = 0; i < D; i++)
      v_[i] = v[i] + _v[i];
    return v_;
  }

  Vector<T, D> operator*(const Vector<T, D> &_v) const {
    Vector<T, D> v_;
    for (size_t i = 0; i < D; i++)
      v_[i] = v[i] * _v[i];
    return v_;
  }

  Vector<T, D> operator/(const Vector<T, D> &_v) const {
    Vector<T, D> v_;
    for (size_t i = 0; i < D; i++)
      v_[i] = v[i] / _v[i];
    return v_;
  }

  Vector<T, D> operator/=(T f) {
    for (size_t i = 0; i < D; i++)
      v[i] /= f;
    return *this;
  }

  Vector<T, D> operator-=(const Vector<T, D> &_v) {
    for (size_t i = 0; i < D; i++)
      v[i] -= _v[i];
    return *this;
  }

  Vector<T, 2> operator/(T f) const {
    T inv = static_cast<T>(1) / f;
    return Vector<T, 2>(v[0] * inv, v[1] * inv);
  }
  Vector<T, 2> xy(size_t x = 0, size_t y = 1) const {
    return Vector<T, 2>(v[x], v[y]);
  }

  Vector<T, 2> floatXY(size_t x = 0, size_t y = 1) const {
    return Vector<T, 2>(static_cast<float>(v[x]), static_cast<float>(v[y]));
  }

  Vector<T, 3> floatXYZ(size_t x = 0, size_t y = 1, size_t z = 2) {
    return Vector<T, 3>(static_cast<float>(v[x]), static_cast<float>(v[y]),
                        static_cast<float>(v[z]));
  }

  T max() const {
    T m = v[0];
    for (size_t i = 1; i < D; i++)
      m = std::max(m, v[i]);
    return m;
  }

  float length2() const {
    float sum = 0.f;
    for (size_t i = 0; i < size; i++)
      sum += SQR(v[i]);
    return sum;
  }
  float length() const { return std::sqrt(length2()); }

  Vector<T, D> normalized() const {
    float d = length();
    Vector<T, D> r;
    for (size_t i = 0; i < size; i++)
      r[i] = v[i] / d;
    return r;
  }

  Vector<T, 2> right() const { return Vector<T, 2>(v[1], -v[0]); }
  Vector<T, 2> left() const { return Vector<T, 2>(-v[1], v[0]); }
  friend std::ostream &operator<<(std::ostream &os, const Vector &v) {
    os << "Vector[<" << D << ">]";
    for (size_t i = 0; i < v.size; i++)
      os << v[i] << " ";
    os << std::endl;
    return os;
  }

  size_t size;
  T v[D];
};

template <typename T, size_t D>
inline Vector<T, D> operator*(float f, const Vector<T, D> &v) {
  return v * f;
}

typedef Vector<float, 2> vec2f;
typedef Vector<int, 2> ivec2;
typedef Vector<uint, 2> uivec2;
typedef Vector<int, 3> ivec3;
typedef Vector<uint, 3> uivec3;
typedef Vector<int, 4> ivec4;
typedef Vector<uint, 4> uivec4;

/* round
 * @v **[in]** vector
 * @return a vector with ceil applied to all components
 */
Vector<int, 3> ceil(const Vector3 &v);
/* round
 * @v **[in]** vector
 * @return a vector with floor applied to all components
 */
Vector<int, 3> floor(const Vector3 &v);

Vector<int, 3> min(Vector<int, 3> a, Vector<int, 3> b);
Vector<int, 3> max(Vector<int, 3> a, Vector<int, 3> b);

/* round
 * @v **[in]** vector
 * @return a vector with ceil applied to all components
 */
Vector<int, 2> ceil(const Vector2 &v);
/* round
 * @v **[in]** vector
 * @return a vector with floor applied to all components
 */
Vector<int, 2> floor(const Vector2 &v);

Vector<int, 2> min(Vector<int, 2> a, Vector<int, 2> b);
Vector<int, 2> max(Vector<int, 2> a, Vector<int, 2> b);

template <typename T, size_t D> T normalize(Vector<T, D> &v) {
  float d = v.length();
  for (size_t i = 0; i < D; i++)
    v[i] = v[i] / d;
}

} // ponos namespace

#endif
