#include "geometry/quaternion.h"

namespace ponos {

	Quaternion::Quaternion() {
		v = Vector3(0.f, 0.f, 0.f);
		w = 1.f;
	}

	Quaternion::Quaternion(Vector3 _v, float _w)
		: v(_v), w(_w) {}

	Quaternion::Quaternion(const Transform &t)
		: Quaternion(t.matrix()) {}

	Quaternion::Quaternion(const Matrix4x4 &m) {
		fromMatrix(m);
	}

} // ponos namespace
