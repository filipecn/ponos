#ifndef HELIOS_MATERIALS_MATTE_H
#define HELIOS_MATERIALS_MATTE_H

#include <helios/core/material.h>
#include <helios/core/spectrum.h>
#include <helios/core/texture.h>

namespace helios {

/// represents a purely diffuse surface
class MatteMaterial : public Material {
public:
  /// \param kd diffuse reflection
  /// \param sigma scalar roughness
  /// \param bump **[optional]** used to compute shading normals
  MatteMaterial(const std::shared_ptr<Texture<Spectrum>> &kd,
                const std::shared_ptr<Texture<real_t>> &sigma,
                const std::shared_ptr<Texture<real_t>> &bump);
  /// Determines the reflective properties at the point and initializes the
  /// SurfaceInteraction::bsdf and SurfaceInteraction::bssrdf members
  /// \param si **[in/out]** surface interaction instance
  /// \param arena used to allocate memory for BSDFs and BSSRDFs
  /// \param mode indicates whether the surface intersection was found along a
  /// path starting from camera or light
  /// \param allowMultipleLobes indicates whether the material should use BxDFs
  /// that aggregate multiple types of scattering into a single BxDF
  void computeScatteringFunctions(SurfaceInteraction *si,
                                  ponos::MemoryArena &arena, TransportMode mode,
                                  bool allowMultipleLobes) const override;

private:
  std::shared_ptr<Texture<Spectrum>> Kd;    //!< diffuse reflection
  std::shared_ptr<Texture<real_t>> sigma;   //!< scalar roughness
  std::shared_ptr<Texture<real_t>> bumpMap; //!< used in shading normals
};

} // namespace helios

#endif
