#include "materials/matte.h"

namespace helios {

    BSDF* MatteMaterial::getBSDF(const DifferentialGeometry& dgGeom,
                                 const DifferentialGeometry& dgShading,
                                 ponos::MemoryArena& arena) const {
      // Allocate BSDF
      DifferentialGeometry dgs;
      if(bumpMap)
        bump(bumpMap, dgGeom, dgShading, &dgs);
      else dgs = dgShading;
      BSDF *bsdf = BSDF_ALLOC(arena, BSDF)(dgs, dgGeom.nn);
      // Evaluate
      Spectrum r = kd_->evaluate(dgs).clamp();
      float sig = ponos::clamp(sigma->evaluate(dgs), 0.f, 90.f);
      if(sig == 0.)
        bsdf->add(BSDF_ALLOC(arena, Lambertian)(r));
      else
        bsdf->add(BSDF_ALLOC(arena, OrenNayar)(r, sig));
      return bsdf;
    }
} // helios namespace
