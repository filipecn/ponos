#ifndef PONOS_GEOMETRY_QUATERNION_H
#define PONOS_GEOMETRY_QUATERNION_H

#include <ponos/geometry/matrix.h>
#include <ponos/geometry/transform.h>
#include <ponos/geometry/vector.h>

namespace ponos {

class Quaternion {
public:
  Quaternion();
  Quaternion(vec3 _v, real_t _w);
  Quaternion(const Transform &t);
  Quaternion(const mat4 &m);
  Quaternion operator+(const Quaternion &q) const;
  Quaternion &operator+=(const Quaternion &q);
  Quaternion operator-(const Quaternion &q) const;
  Quaternion operator-() const;
  Quaternion &operator-=(const Quaternion &q);
  Quaternion operator/(real_t d) const;
  Quaternion operator*(real_t d) const;
  Quaternion operator*(const Quaternion &q) const;
  bool operator==(const Quaternion &q);
  void fromAxisAndAngle(const vec3 &_v, real_t angle);
  void fromMatrix(const mat4 &m);
  mat3 toRotationMatrix() const;
  Transform toTransform() const;

  friend std::ostream &operator<<(std::ostream &os, const Quaternion &q) {
    os << "[quat]\n" << q.v << " w = " << q.w << std::endl;
    ;
    return os;
  }

  vec3 v;
  real_t w;
};

typedef Quaternion quat;

Quaternion operator*(real_t f, const Quaternion &q);

real_t dot(const Quaternion &q1, const Quaternion &q2);

Quaternion normalize(const Quaternion &q);

Quaternion operator*(vec3 v, const Quaternion &q);

Quaternion slerp(real_t t, const Quaternion &q1, const Quaternion &q2);

} // namespace ponos

#endif
