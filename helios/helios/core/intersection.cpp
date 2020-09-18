#include "core/intersection.h"

namespace helios {

    BSDF* Intersection::getBSDF(const RayDifferential& ray, ponos::MemoryArena &arena) const {
      //dg.computeDifferentials(ray);
      BSDF *bsdf = nullptr;// = primitive->getBSDF(dg, o2w, arena);
      return bsdf;
    }

    //BSSRDF* Intersection::getBSSRDF(const RayDifferential& ray, ponos::MemoryArena &arena) const {
    //  dg.computeDifferentials(ray);
    //  BSSRDF *bssrdf = primitive->getBSSRDF(dg, o2w, arena);
    //  return bssrdf;
    //}

} // helios namespace
