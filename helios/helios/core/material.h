#ifndef HELIOS_CORE_MATERIAL_H
#define HELIOS_CORE_MATERIAL_H

#include <helios/core/interaction.h>
#include <ponos/storage/memory.h>

namespace helios {

/// Material Interface. Base class for Material implementations
class Material {
public:
  Material();
  virtual ~Material();
  /// Determines the reflective properties at the point and initializes the
  /// SurfaceInteraction::bsdf and SurfaceInteraction::bssrdf members
  /// \param si **[in/out]** surface interaction instance
  /// \param arena used to allocate memory for BSDFs and BSSRDFs
  /// \param mode indicates whether the surface intersection was found along a
  /// path starting from camera or light
  /// \param allowMultipleLobes indicates whether the material should use BxDFs
  /// that aggregate multiple types of scattering into a single BxDF
  virtual void computeScatteringFunctions(SurfaceInteraction *si,
                                          ponos::MemoryArena &arena,
                                          TransportMode mode,
                                          bool allowMultipleLobes) const = 0;
  /// Computes shading normals based on a displaced function represented by a
  /// texture (bump mapping).
  /// \param d displacement field
  /// \param si surface interaction containing the normals
  void bump(const std::shared_ptr<Texture<real_t>> &d, SurfaceInteraction *si);
};

} // namespace helios

#endif
