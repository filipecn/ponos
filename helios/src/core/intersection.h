#ifndef HELIOS_CORE_INTERSECTION_H
#define HELIOS_CORE_INTERSECTION_H

#include "core/bsdf.h"
#include "core/differential_geometry.h"
#include "geometry/h_ray.h"

#include <ponos.h>

namespace helios {

	class Primitive;

	struct Intersection {
		DifferentialGeometry dg;
		const Primitive *primitive;
		ponos::Transform w2o, o2w;
		uint32_t shapeId, primitiveId;
		float rayEpsilon;

    BSDF* getBSDF(const RayDifferential& ray, ponos::MemoryArena &arena) const;
    //BSSRDF* getBSSRDF(const RayDifferential& ray, ponos::MemoryArena &arena) const;
	};

} // helios namespace

#endif
