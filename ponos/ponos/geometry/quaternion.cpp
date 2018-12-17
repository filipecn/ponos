#include <ponos/geometry/quaternion.h>

namespace ponos {

Quaternion::Quaternion() {
  v = vec3(0.f, 0.f, 0.f);
  w = 1.f;
}

Quaternion::Quaternion(vec3 _v, float _w) : v(_v), w(_w) {}

Quaternion::Quaternion(const Transform &t) : Quaternion(t.matrix()) {}

Quaternion::Quaternion(const mat4 &m) { fromMatrix(m); }

} // ponos namespace
