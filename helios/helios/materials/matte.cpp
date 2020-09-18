#include <helios/materials/matte.h>

namespace helios {

MatteMaterial::MatteMaterial(const std::shared_ptr<Texture<Spectrum>> &kd,
                             const std::shared_ptr<Texture<real_t>> &sigma,
                             const std::shared_ptr<Texture<real_t>> &bumpMap)
    Kd(kd),
    sigma(sigma), bumpMap(bumpMap) {}

void MatteMaterial::computeScatteringFunctions(SurfaceInteraction *si,
                                               ponos::MemoryArena &arena,
                                               TransportMode mode,
                                               bool allowMultipleLobes) const {
  // perform bump mapping if present
  if (bumpMap)
    bump(bumpMap, si);
  // evaluate textures for material and allocate BRDF
  si->bsdf = ARENA_ALLOC(arena, BSDF)(*si);
  Spectrum r = Kd->evaluate(*si).clamp();
  real_t sig = ponos::clamp(sigma->evaluate(*si), 0, 90);
  if (!r.isBlack()) {
    if (sig == 0)
      si->bsdf->add(ARENA_ALLOC(arena, LambertianReflection)(r));
    else
      si->bsdf->add(ARENA_ALLOC(arena, OrenNayar)(r, sig));
  }
}

} // namespace helios
