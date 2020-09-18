#include <helios/core/bsdf.h>

namespace helios {

BSDF::BSDF(const SurfaceInteraction &si, real_t eta)
    : eta(eta), ns(si.shading.n), ng(si.n),
      ss(ponos::normalize(si.shading.dpdu)), ts(ponos::cross(ns, ss)) {}

void BSDF::add(BxDF *b) { bxdfs[nBxDFs++] = b; }

int BSDF::numComponents(BxDF::Type types) const {}

ponos::vec3 BSDF::worldToLocal(const ponos::vec3 &v) const {
  return ponos::vec3(ponos::dot(v, ss), ponos::dot(v, ts), ponos::dot(v, ns));
}

ponos::vec3 BSDF::localToWorld(const ponos::vec3 &v) const {
  return ponos::vec3(ss.x * v.x + ts.x * v.y + ns.x * v.z,
                     ss.y * v.x + ts.y * v.y + ns.y * v.z,
                     ss.z * v.x + ts.z * v.y + ns.z * v.z);
}

Spectrum BSDF::f(const ponos::vec3 &woW, const ponos::vec3 &wiW,
                 BxDF::Type types) const {
  ponos::vec3 wi = worldToLocal(wiW), wo = worldToLocal(woW);
  bool reflect = ponos::dot(wiW, ng) * ponos::dot(woW, ng) > 0;
  Spectrum s(0.f);
  for (int i = 0; i < nBxDFs; ++i)
    if (bxdfs[i]->matchesFlags(types) &&
        ((reflect && (bxdfs[i]->type & BxDF::Type::BSDF_REFLECTION)) ||
         (!reflect && (bxdfs[i]->type & BxDF::Type::BSDF_TRANSMISSION))))
      s += bxdfs[i]->f(wo, wi);
  return s;
}

} // namespace helios
