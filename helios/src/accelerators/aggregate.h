#ifndef HELIOS_ACCELERATORS_AGGREGATE_H
#define HELIOS_ACCELERATORS_AGGREGATE_H

#include "core/primitive.h"

namespace helios {

	class Aggregate : public Primitive {
  	public:
	 		Aggregate() {}
			virtual ~Aggregate() {}

			// const AreaLight *getAreaLight() const override = delete;
			// virtual BSDF *getBSDF(const DifferentialGeometry &gd, const ponos::Transform &o2w, MemoryArena &arena) const = delete;
			// virtual BSSRDF *getBSSRDF(const DifferentialGeometry &gd, const Transform &o2w, MemoryArena &arena) const = delete;
	};

} // helios namespace

#endif

