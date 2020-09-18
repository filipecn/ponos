#include <helios/materials/plastic.h>

namespace helios {

PlasticMaterial::PlasticMaterial(
    const std::shared_ptr<Texture<Spectrum>> &kd,
    const std::shared_ptr<Texture<Spectrum>> &ks,
    const std::shared_ptr<Texture<real_t>> &roughness,
    const std::shared_ptr<Texture<real_t>> &bump, bool remapRoughness)
    : Kd(kd), Ks(ks), roughness(roughness), bumpMap(bumpMap),
      remapRoughness(remapRoughness) {}

void PlasticMaterial::computeScatteringFunctions(
    SurfaceInteraction *si, ponos::MemoryArena &arena, TransportMode mode,
    bool allowMultipleLobes) const {
  // perform bump mapping if present
  if (bumpMap)
    bump(bumpMap, si);
  si->bsdf = ARENA_ALLOC(arena, BSDF)(*si);
  // initialize diffuse component
  Spectrum kd = Kd->evaluate(*si).clamp();
  if (!kd.isBlack())
    si->bsdf->add(ARENA_ALLOC(arena, LambertianReflection)(kd));
  // initialize specular component
  Spectrum ks = Ks->evaluate(*si).clamp();
  if (!kd.isBlack()) {
    Fresnel *fresnel = ARENA_ALLOC(arena, FresnelDielectric)(1.f, 1.5f);
    // create microfacet distribution
    real_t rough = roughness->evaluate(*si);
    if (remapRoughness)
      rough = TrowbridgeReitzDistribution::roughnessToAlpha(rough);
    MicrofacetDistribution *distrib =
        ARENA_ALLOC(arena, TrowbridgeReitzDistribution)(rough, rough);
    BxDF *spec = ARENA_ALLOC(arena, MicrofacetReflection)(ks, distrib, fresnel);
    si->bsdf->add(spec);
  }
}

} // namespace helios
