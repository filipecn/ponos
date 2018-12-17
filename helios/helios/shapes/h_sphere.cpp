#include <helios/shapes/h_sphere.h>

using namespace ponos;

namespace helios {

HSphere::HSphere(const HTransform *o2w, const HTransform *w2o, bool ro,
                 real_t rad, real_t z0, real_t z1, real_t pm)
    : Shape(o2w, w2o, ro) {
  zmin = clamp(std::min(z0, z1), -radius, radius);
  radius = rad;
  zmax = clamp(std::max(z0, z1), -radius, radius);
  thetaMin = acosf(clamp<real_t>(zmin / radius, -1, 1));
  thetaMax = acosf(clamp<real_t>(zmin / radius, -1, 1));
  thetaMax = TO_RADIANS(clamp<real_t>(pm, 0, 360));
}

bounds3f HSphere::objectBound() const {
  return bounds3f(point3(-radius, -radius, zmin), point3(radius, radius, zmax));
}

bool HSphere::intersect(const HRay &r, real_t *tHit, SurfaceInteraction *isect,
                        bool testAlphaTexture) const {
  real_t phi;
  point3 phit;
  // transform HRay to object space
  vec3f oErr, dErr;
  HRay ray = (*this->worldToObject)(r, &oErr, &dErr);
  //    initialize efloat ray coordinate valyes
  EFloat ox(ray.o.x, oErr.x), oy(ray.o.y, oErr.y), oz(ray.o.z, oErr.z);
  EFloat dx(ray.d.x, dErr.x), dy(ray.d.y, dErr.y), dz(ray.d.z, dErr.z);
  // compute quadritic HSphere coefficients
  EFloat a = dx * dx + dy * dy + dz * dz;
  EFloat b = 2 * (dx * ox + dy * oy + dz * oz);
  EFloat c = ox * ox + oy * oy + oz * oz - EFloat(radius) * EFloat(radius);
  // solve quadritic equation for t values
  EFloat t0, t1;
  if (!solve_quadratic(a, b, c, &t0, &t1))
    return false;
  // check quadric shape t0 and t1 for nearest intersection
  if (t0.upperBound() > ray.max_t || t1.lowerBound() <= 0)
    return false;
  EFloat thit = t0;
  if (thit.lowerBound() <= 0) {
    thit = t1;
    if (thit.upperBound() > ray.max_t)
      return false;
  }
  // compute HSphere hit position and phi
  phit = ray((real_t)thit);
  // refine sphere intersection point
  phit *= radius / distance(phit, point3f(0));
  if (phit.x == 0 && phit.y == 0)
    phit.x = 1e-5f * radius;
  phi = std::atan2(phit.y, phit.x);
  if (phi < 0.)
    phi += 2 * Constants::pi;
  // test HSphere intersection against clipping parameters
  if ((zmin > -radius && phit.z < zmin) || (zmax < radius && phit.z > zmax) ||
      phi > phiMax) {
    if (thit == t1)
      return false;
    if (t1.upperBound() > ray.max_t)
      return false;
    thit = t1;
    // compute HSphere hit position and phi
    phit = ray((real_t)thit);
    // refine sphere intersection point
    phit *= radius / distance(phit, point3f(0));
    if (phit.x == 0 && phit.y == 0)
      phit.x = 1e-5f * radius;
    phi = std::atan2(phit.y, phit.x);
    if (phi < 0.)
      phi += 2 * Constants::pi;
    if ((zmin > -radius && phit.z < zmin) || (zmax < radius && phit.z > zmax) ||
        phi > phiMax)
      return false;
  }
  // find parametric representation of HSphere hit
  real_t u = phi / phiMax;
  real_t theta = std::acos(clamp<real_t>(phit.z / radius, -1, 1));
  real_t v = (theta - thetaMin) / (thetaMax - thetaMin);
  // compute HSphere dp/du and dp/dv
  real_t zradius = std::sqrtf(phit.x * phit.x + phit.y * phit.y);
  real_t invradius = 1.f / zradius;
  real_t cosphi = phit.x * invradius;
  real_t sinphi = phit.y * invradius;
  vec3f dpdu(-phiMax * phit.y, phiMax * phit.x, 0);
  vec3f dpdv = (thetaMax - thetaMin) * vec3(phit.z * cosphi, phit.z * sinphi,
                                            -radius * std::sin(theta));
  // compute HSphere dn/du and dn/dv
  vec3 d2Pduu = -phiMax * phiMax * vec3f(phit.x, phit.y, 0);
  vec3 d2Pduv =
      (thetaMax - thetaMin) * phit.z * phiMax * vec3f(-sinphi, cosphi, 0.f);
  vec3f d2Pdvv = -(thetaMax - thetaMin) * (thetaMax - thetaMin) *
                 vec3f(phit.x, phit.y, phit.z);
  // compute coefficients for fundamental forms
  real_t E = dot(dpdu, dpdu);
  real_t F = dot(dpdu, dpdv);
  real_t G = dot(dpdv, dpdv);
  vec3f N = normalize(cross(dpdu, dpdv));
  real_t e = dot(N, d2Pduu);
  real_t f = dot(N, d2Pduv);
  real_t g = dot(N, d2Pdvv);
  // compute dndu and dndv from fundamental form coefficients
  real_t invEFG2 = 1 / (E * G - F * F);
  normal3f dndu((f * F - e * G) * invEFG2 * dpdu +
                (e * F - f * E) * invEFG2 * dpdv);
  normal3f dndv((g * F - f * G) * invEFG2 * dpdu +
                (f * F - g * E) * invEFG2 * dpdv);
  // compute error bounds for sphere intersection
  vec3f pError = gammaBound(5) * abs((vec3f)phit);
  // initialize SurfaceInteraction from parametric information
  *isect = (*objectToWorld)(SurfaceInteraction(phit, pError, point2f(u, v),
                                               -ray.d, dpdu, dpdv, dndu, dndv,
                                               ray.time, this));
  // update tHit for quadric intersection
  *tHit = (real_t)(thit);
  return true;
}

bool HSphere::intersectP(const HRay &r, bool testAlphaTexture) const {
  real_t phi;
  point3 phit;
  // transform HRay to object space
  vec3f oErr, dErr;
  HRay ray = (*this->worldToObject)(r, &oErr, &dErr);
  //    initialize efloat ray coordinate valyes
  EFloat ox(ray.o.x, oErr.x), oy(ray.o.y, oErr.y), oz(ray.o.z, oErr.z);
  EFloat dx(ray.d.x, dErr.x), dy(ray.d.y, dErr.y), dz(ray.d.z, dErr.z);
  // compute quadritic HSphere coefficients
  EFloat a = dx * dx + dy * dy + dz * dz;
  EFloat b = 2 * (dx * ox + dy * oy + dz * oz);
  EFloat c = ox * ox + oy * oy + oz * oz - EFloat(radius) * EFloat(radius);
  // solve quadritic equation for t values
  EFloat t0, t1;
  if (!solve_quadratic(a, b, c, &t0, &t1))
    return false;
  // check quadric shape t0 and t1 for nearest intersection
  if (t0.upperBound() > ray.max_t || t1.lowerBound() <= 0)
    return false;
  EFloat thit = t0;
  if (thit.lowerBound() <= 0) {
    thit = t1;
    if (thit.upperBound() > ray.max_t)
      return false;
  }
  // compute HSphere hit position and phi
  phit = ray((real_t)thit);
  // refine sphere intersection point
  phit *= radius / distance(phit, point3f(0));
  if (phit.x == 0 && phit.y == 0)
    phit.x = 1e-5f * radius;
  phi = std::atan2(phit.y, phit.x);
  if (phi < 0.)
    phi += 2 * Constants::pi;
  // test HSphere intersection against clipping parameters
  if ((zmin > -radius && phit.z < zmin) || (zmax < radius && phit.z > zmax) ||
      phi > phiMax) {
    if (thit == t1)
      return false;
    if (t1.upperBound() > ray.max_t)
      return false;
    thit = t1;
    // compute HSphere hit position and phi
    phit = ray((real_t)thit);
    // refine sphere intersection point
    phit *= radius / distance(phit, point3f(0));
    if (phit.x == 0 && phit.y == 0)
      phit.x = 1e-5f * radius;
    phi = std::atan2(phit.y, phit.x);
    if (phi < 0.)
      phi += 2 * Constants::pi;
    if ((zmin > -radius && phit.z < zmin) || (zmax < radius && phit.z > zmax) ||
        phi > phiMax)
      return false;
  }

  return true;
}

real_t HSphere::surfaceArea() const { return phiMax * radius * (zmax - zmin); }

} // namespace helios
