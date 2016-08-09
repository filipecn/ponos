#include "core/differential_geometry.h"

#include <ponos.h>

namespace helios {

	class Primitive;

	struct Intersection {
		DifferentialGeometry dg;
		const Primitive *primitive;
		ponos::Transform w2o, o2w;
		uint32_t shapeId, primitiveId;
		float rayEpsilon;
	};

} // helios namespace
