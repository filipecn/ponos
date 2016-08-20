#include "geometry/normal.h"
#include "geometry/vector.h"

namespace ponos {

  Normal::Normal(float _x, float _y, float _z)
    : x(_x), y(_y), z(_z) {}

  Normal::Normal(const Vector3& v)
    : x(v.x), y(v.y), z(v.z) {}

} // ponos namespacec
