#include <ponos/geometry/quaternion.h>

namespace ponos {

Quaternion::Quaternion() {
  v = vec3(0.f, 0.f, 0.f);
  w = 1.f;
}

Quaternion::Quaternion(vec3 _v, float _w) : v(_v), w(_w) {}

Quaternion::Quaternion(const Transform &t) : Quaternion(t.matrix()) {}

Quaternion::Quaternion(const mat4 &m) { fromMatrix(m); }

Quaternion Quaternion::operator+(const Quaternion &q) const {
  return Quaternion(v + q.v, w + q.w);
}

Quaternion &Quaternion::operator+=(const Quaternion &q) {
  v += q.v;
  w += q.w;
  return *this;
}

Quaternion Quaternion::operator-(const Quaternion &q) const {
  return Quaternion(v - q.v, w - q.w);
}

Quaternion Quaternion::operator-() const { return Quaternion(-v, -w); }

Quaternion &Quaternion::operator-=(const Quaternion &q) {
  v -= q.v;
  w -= q.w;
  return *this;
}

Quaternion Quaternion::operator/(real_t d) const {
  return Quaternion(v / d, w / d);
}

Quaternion Quaternion::operator*(real_t d) const {
  return Quaternion(v * d, w * d);
}

Quaternion Quaternion::operator*(const Quaternion &q) const {
  return Quaternion(w * q.v + q.w * v + cross(v, q.v), w * q.w - dot(v, q.v));
}

bool Quaternion::operator==(const Quaternion &q) {
  return Check::is_equal(w, q.w) && v == q.v;
}

void Quaternion::fromAxisAndAngle(const vec3 &_v, real_t angle) {
  real_t theta = RADIANS(angle / 2);
  v = _v * sinf(theta);
  w = cosf(theta);
}

void Quaternion::fromMatrix(const mat4 &m) {
  // extracted from Shoemake, 1991
  const size_t X = 0;
  const size_t Y = 1;
  const size_t Z = 2;
  const size_t W = 3;
  real_t tr, s;
  tr = m[X][X] + m[Y][Y] + m[Z][Z];
  if (tr >= 0.f) {
    s = sqrtf(tr + m[W][W]);
    w = s * .5f;
    s = .5f / s;
    v.x = (m[Z][Y] - m[Y][Z]) * s;
    v.y = (m[X][Z] - m[Z][X]) * s;
    v.z = (m[Y][X] - m[X][Y]) * s;
  } else {
    size_t h = X;
    if (m[Y][Y] > m[X][X])
      h = Y;
    if (m[Z][Z] > m[h][h])
      h = Z;
    switch (h) {
#define case_macro(i, j, k, I, J, K)                                           \
  case I:                                                                      \
    s = sqrtf((m[I][I] - (m[J][J] + m[K][K])) + m[W][W]);              \
    v.i = s * .5f;                                                             \
    s = .5f / s;                                                               \
    v.j = (m[I][J] + m[J][I]) * s;                                         \
    v.k = (m[K][I] + m[I][K]) * s;                                         \
    w = (m[K][J] + m[J][K]) * s;                                           \
    break;
    case_macro(x, y, z, X, Y, Z);
    case_macro(y, z, x, Y, Z, X);
    case_macro(z, x, y, Z, X, Y);
#undef case_macro
    }
  }
  if (m[W][W] != 1.0) {
    s = 1.0 / sqrtf(m[W][W]);
    w *= s;
    v *= s;
  }
}

mat3 Quaternion::toRotationMatrix() const {
  return mat3(1.f - 2.f * (v.y * v.y + v.z * v.z), 2.f * (v.x * v.y - v.z * w),
              2.f * (v.x * v.z + v.y * w), 2.f * (v.x * v.y + v.z * w),
              1.f - 2.f * (v.x * v.x + v.z * v.z), 2.f * (v.y * v.z - v.x * w),
              2.f * (v.x * v.z - v.y * w), 2.f * (v.y * v.z + v.x * w),
              1.f - 2.f * (v.x * v.x + v.y * v.y));
}

Transform Quaternion::toTransform() const {
  real_t m[4][4];
  m[0][0] = 1.f - 2.f * (v.y * v.y + v.z * v.z);
  m[1][0] = 2.f * (v.x * v.y + v.z * w);
  m[2][0] = 2.f * (v.x * v.z - v.y * w);
  m[3][0] = 0.f;

  m[0][1] = 2.f * (v.x * v.y - v.z * w);
  m[1][1] = 1.f - 2.f * (v.x * v.x + v.z * v.z);
  m[2][1] = 2.f * (v.y * v.z + v.x * w);
  m[3][1] = 0.f;

  m[0][2] = 2.f * (v.x * v.z + v.y * w);
  m[1][2] = 2.f * (v.y * v.z - v.x * w);
  m[2][2] = 1.f - 2.f * (v.x * v.x + v.y * v.y);
  m[3][2] = 0.f;

  m[0][3] = 0.f;
  m[1][3] = 0.f;
  m[2][3] = 0.f;
  m[3][3] = 1.f;

  return Transform(m);
}

Quaternion operator*(real_t f, const Quaternion &q) { return q * f; }

real_t dot(const Quaternion &q1, const Quaternion &q2) {
  return dot(q1.v, q2.v) + q1.w * q2.w;
}

Quaternion normalize(const Quaternion &q) { return q / sqrtf(dot(q, q)); }

Quaternion operator*(vec3 v, const Quaternion &q) {
  return Quaternion(v, 0) * q;
}

Quaternion slerp(real_t t, const Quaternion &q1, const Quaternion &q2) {
  real_t cosTheta = dot(q1, q2);
  if (cosTheta > .9995f)
    return normalize((1.f - t) * q1 + t * q2);
  real_t theta = acosf(clamp(cosTheta, -1.f, 1.f));
  real_t thetap = theta * t;
  Quaternion qperp = normalize(q2 - q1 * cosTheta);
  return q1 * cosf(thetap) + qperp * sinf(thetap);
}

} // namespace ponos
