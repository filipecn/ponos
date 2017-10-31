#ifndef HELIOS_CORE_MATERIAL_H
#define HELIOS_CORE_MATERIAL_H

#include "core/bsdf.h"
#include "core/differential_geometry.h"
#include "core/texture.h"

#include <ponos.h>

#include <memory>

namespace helios {

	class Material {
  	public:
	 		Material() {}
			virtual ~Material() {}
      virtual BSDF* getBSDF(const DifferentialGeometry& dgGeom,
                            const DifferentialGeometry& dgShading,
                            ponos::MemoryArena& arena) const = 0;
      virtual BSDF* getBSSDF(const DifferentialGeometry& dgGeom,
                             const DifferentialGeometry& dgShading,
                             ponos::MemoryArena& arena) const {
                               return nullptr;
                             }
      void bump(const std::shared_ptr<Texture<float> > &d,
                const DifferentialGeometry &dgGeom,
                const DifferentialGeometry &dgs,
                DifferentialGeometry *dgBump);
	};

} // helios namespace

#endif
