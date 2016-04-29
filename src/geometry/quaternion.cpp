#include "geometry/quaternion.h"

namespace ponos {

  Quaternion::Quaternion() {
    v = Vector3(0.f, 0.f, 0.f);
    w = 1.f;
  }

  Quaternion::Quaternion(Vector3 _v, float _w)
    : v(_v), w(_w) {}

};
