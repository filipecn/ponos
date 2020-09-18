#include <helios/core/material.h>

namespace helios {

void Material::bump(const std::shared_ptr<Texture<real_t>> &d,
                    SurfaceInteraction *si) {
  // compute offset positions and evaluate displacement texture
  SurfaceInteraction siEval = *si;
  // shift siEval du in the u direction
  real_t du = .5f * (std::abs(si->dudx) + std::abs(si->dudy));
  if (du == 0)
    du = .01f;
  siEval.p = si->p + du * si->shading.dpdu;
  siEval.uv = si->uv + ponos::vec2(du, 0.f);
  siEval.n = ponos::normalize(
      (ponos::normal3)ponos::cross(si->shading.dpdu, si->shading.dpdv) +
      du * si->dndu);
  real_t uDisplace = d->evaluate(siEval);
  // shift siEval dv in the v direction
  real_t dv = .5f * (std::abs(si->dvdx) + std::abs(si->dvdy));
  if (dv == 0)
    dv = .01f;
  siEval.p = si->p + dv * si->shading.dpdu;
  siEval.uv = si->uv + ponos::vec2(dv, 0.f);
  siEval.n = ponos::normalize(
      (ponos::normal3)ponos::cross(si->shading.dpdu, si->shading.dpdv) +
      dv * si->dndv);
  real_t vDisplace = d->evaluate(siEval);
  real_t displace = d->evaluate(*si);
  // compute bump-mapped differential geometry
  ponos::vec3 dpdu = si->shading.dpdu +
                     (uDisplace - displace) / du * ponos::vec3(si->shading.n) +
                     displace * ponos::vec3(si->shading.dndu);
  ponos::vec3 dpdv = si->shading.dpdv +
                     (vDisplace - displace) / dv * ponos::vec3(si->shading.n) +
                     displace * ponos::vec3(si->shading.dndv);
  si->setShadingGeometry(dpdu, dpdv, si->shading.dndu, si->shading.dndv, false);
}

} // namespace helios
