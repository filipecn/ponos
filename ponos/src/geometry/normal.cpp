#include "geometry/normal.h"
#include "geometry/vector.h"

namespace ponos {

  Normal2D::Normal2D(float _x, float _y)
    : x(_x), y(_y) {}

  Normal2D::Normal2D(const Vector2& v)
    : x(v.x), y(v.y) {}

	Normal2D::operator Vector2() const {
		return Vector2(x, y);
	}

  Normal::Normal(float _x, float _y, float _z)
    : x(_x), y(_y), z(_z) {}

  Normal::Normal(const Vector3& v)
    : x(v.x), y(v.y), z(v.z) {}

	Normal::operator Vector3() const {
		return Vector3(x, y, z);
	}

} // ponos namespacec
