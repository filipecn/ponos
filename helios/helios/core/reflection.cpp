#include <helios/core/reflection.h>

namespace helios {

real_t ShadingCoordinateSystem::cosTheta(const ponos::vec3 &w) { return w.z; }

real_t ShadingCoordinateSystem::cos2Theta(const ponos::vec3 &w) {
  return w.z * w.z;
}

real_t ShadingCoordinateSystem::absCosTheta(const ponos::vec3 &w) {
  return std::abs(w.z);
}

real_t ShadingCoordinateSystem::sinTheta(const ponos::vec3 &w) {
  return std::sqrt(sin2Theta(w));
}

real_t ShadingCoordinateSystem::sin2Theta(const ponos::vec3 &w) {
  return std::max(static_cast<real_t>(0),
                  static_cast<real_t>(1) - cos2Theta(w));
}

real_t ShadingCoordinateSystem::tanTheta(const ponos::vec3 &w) {
  return sinTheta(w) / cosTheta(w);
}

real_t ShadingCoordinateSystem::tan2Theta(const ponos::vec3 &w) {
  return sin2Theta(w) / cos2Theta(w);
}

real_t ShadingCoordinateSystem::cosPhi(const ponos::vec3 &w) {
  real_t _sinTheta = sinTheta(w);
  return (_sinTheta == 0) ? 1 : ponos::clamp(w.x / _sinTheta, -1, 1);
}

real_t ShadingCoordinateSystem::cos2Phi(const ponos::vec3 &w) {
  return cosPhi(w) * cosPhi(w);
}

real_t ShadingCoordinateSystem::sinPhi(const ponos::vec3 &w) {
  real_t _sinTheta = sinTheta(w);
  return (_sinTheta == 0) ? 1 : ponos::clamp(w.y / _sinTheta, -1, 1);
}

real_t ShadingCoordinateSystem::sin2Phi(const ponos::vec3 &w) {
  return sinPhi(w) * sinPhi(w);
}

real_t ShadingCoordinateSystem::cosDPhi(const ponos::vec3 &wa,
                                        const ponos::vec3 &wb) {
  return ponos::clamp(
      (wa.x * wb.x + wa.x * wb.y) /
          std::sqrt((wa.x * wa.x + wa.y * wa.y) * (wb.x * wb.x + wb.y * wb.y)),
      -1, 1);
}

BxDF::BxDF(BxDF::Type type) : type(type) {}

bool BxDF::matchesFlags(BxDF::Type t) const { return (type & t) == type; }

ScaledBxDF::ScaledBxDF(BxDF *bxdf, const Spectrum &scale)
    : BxDF(BxDF::Type(bxdf->type)), bxdf(bxdf), scale(scale) {}

Spectrum ScaledBxDF::f(const ponos::vec3 &wo, const ponos::vec3 &wi) const {
  return scale * bxdf->f(wo, wi);
}

real_t Fresnel::frDielectric(real_t cosThetaI, real_t etaI, real_t etaT) {
  cosThetaI = ponos::clamp(cosThetaI, -1, 1);
  // potentially swap indice of refraction
  bool entering = cosThetaI > 0.f;
  if (!entering) {
    std::swap(etaI, etaT);
    cosThetaI = std::abs(cosThetaI);
  }
  // compute cosThetaT using Snell's law
  real_t sinThetaI =
      std::sqrt(std::max(static_cast<real_t>(0), 1 - cosThetaI * cosThetaI));
  real_t sinThetaT = etaI / etaT * sinThetaI;
  // handle total internal reflection
  if (sinThetaT >= 1)
    return 1;
  real_t cosThetaT =
      std::sqrt(std::max(static_cast<real_t>(0), 1 - sinThetaT * sinThetaT));
  real_t Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                 ((etaT * cosThetaI) + (etaI * cosThetaT));
  real_t Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                 ((etaI * cosThetaI) + (etaT * cosThetaT));
  return (Rparl * Rparl + Rperp * Rperp) / 2;
}

Spectrum Fresnel::frConductor(real_t cosThetaI, const Spectrum &etaI,
                              const Spectrum &etaT, const Spectrum &k) {
  cosThetaI = ponos::clamp(cosThetaI, -1, 1);
  Spectrum eta = etat / etai;
  Spectrum etak = k / etai;

  real_t cosThetaI2 = cosThetaI * cosThetaI;
  real_t sinThetaI2 = 1. - cosThetaI2;
  Spectrum eta2 = eta * eta;
  Spectrum etak2 = etak * etak;

  Spectrum t0 = eta2 - etak2 - sinThetaI2;
  Spectrum a2plusb2 = Sqrt(t0 * t0 + 4 * eta2 * etak2);
  Spectrum t1 = a2plusb2 + cosThetaI2;
  Spectrum a = Sqrt(0.5f * (a2plusb2 + t0));
  Spectrum t2 = (Float)2 * cosThetaI * a;
  Spectrum Rs = (t1 - t2) / (t1 + t2);

  Spectrum t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
  Spectrum t4 = t2 * sinThetaI2;
  Spectrum Rp = Rs * (t3 - t4) / (t3 + t4);

  return 0.5 * (Rp + Rs);
}

FresnelConductor::FresnelConductor(const Spectrum &etaI, const Spectrum &etaT,
                                   const Spectrum &k)
    : etaI(etaI), etaT(etaT), k(k) {}

Spectrum FresnelConductor::evaluate(real_t cosI) const {
  return Fresnel::frConductor(std::abs(cosI), etaI, etaT, k);
}

FresnelDielectric::FresnelDielectric(const real_t &etaI, const real_t &etaT)
    : etaI(etaI), etaT(etaT) {}

Spectrum FresnelDielectric::evaluate(real_t cosI) const {
  return Fresnel::frDielectric(cosI, etaI, etaT);
}

} // namespace helios