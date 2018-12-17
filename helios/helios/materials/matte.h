#ifndef HELIOS_MATERIALS_MATTE_H
#define HELIOS_MATERIALS_MATTE_H

#include "core/bsdf.h"
#include "core/material.h"
#include "core/spectrum.h"
#include "core/texture.h"

namespace helios {

  /* material
   * represents a purely diffuse surface
   */
  class MatteMaterial : public Material {
  public:
    MatteMaterial(Texture<Spectrum>* kd, Texture<float>* sig, Texture<float>* bump) {
      kd_.reset(kd);
      sigma.reset(sig);
      bumpMap.reset(bump);
    }
    BSDF* getBSDF(const DifferentialGeometry& dgGeom,
                  const DifferentialGeometry& dgShading,
                  ponos::MemoryArena& arena) const;
  };

} // helios namespace

#endif
