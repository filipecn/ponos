#include "shapes/h_sphere.h"

using namespace ponos;

namespace helios {

  HSphere::HSphere(const Transform *o2w, const Transform *w2o, bool ro,
                 float rad, float z0, float z1, float pm)
    : Shape(o2w, w2o, ro) {
      zmin = clamp(std::min(z0, z1), -radius, radius);
    radius = rad;
    zmax = clamp(std::max(z0, z1), -radius, radius);
    thetaMin = acosf(clamp(zmin / radius, -1.f, 1.f));
    thetaMax = acosf(clamp(zmin / radius, -1.f, 1.f));
    thetaMax = TO_RADIANS(clamp(pm, 0.f, 360.f));
  }

  BBox HSphere::objectBound() const {
    return BBox(Point3(-radius, -radius, zmin), Point3(radius, radius, zmax));
  }

  bool HSphere::intersect(const HRay &r, float *tHit, float *HRayEpsilon,
                         DifferentialGeometry *dg) const {
    float phi;
    Point3 phit;
    // transform HRay to object space
    HRay HRay;
    (*worldToObject)(r, &HRay);
    // compute quadritic HSphere coefficients
    float A = HRay.d.x * HRay.d.x + HRay.d.y * HRay.d.y + HRay.d.z * HRay.d.z;
    float B = 2 * (HRay.d.x * HRay.o.x + HRay.d.y * HRay.o.y + HRay.d.z * HRay.o.z);
    float C = HRay.o.x * HRay.o.x + HRay.o.y * HRay.o.y + HRay.o.z * HRay.o.z - radius * radius;
    // solve quadritic equation for t values
    float t0, t1;
    if(!solve_quadratic(A, B, C, t0, t1))
      return false;
    if(t0 > HRay.max_t || t1 < HRay.min_t)
      return false;
    float thit = t0;
    if(t0 < HRay.min_t) {
      thit = t1;
      if(thit > HRay.max_t)
        return false;
    }
    // compute HSphere hit position and phi
    phit = HRay(thit);
    if(IS_ZERO(phit.x) && IS_ZERO(phit.y))
      phit.x = 1e-5f * radius;
    phi = atan2f(phit.y, phit.x);
    if(phi < 0.) phi += 2.f * PI;
    // test HSphere intersection against clipping parameters
    if((zmin > -radius && phit.z < zmin) ||
       (zmax <  radius && phit.z > zmax) || phi > phiMax) {
      if(thit == t1) return false;
      if(t1 > HRay.max_t) return false;
      thit = t1;
      // compute HSphere hit position and phi
      phit = HRay(thit);
      if(IS_ZERO(phit.x) && IS_ZERO(phit.y))
        phit.x = 1e-5f * radius;
      phi = atan2f(phit.y, phit.x);
      if(phi < 0.) phi += 2.f * PI;
      if((zmin > -radius && phit.z < zmin) ||
         (zmax <  radius && phit.z > zmax) || phi > phiMax)
          return false;
    }
    // find parametric representation of HSphere hit
    float u = phi / phiMax;
    float theta = acosf(clamp(phit.z / radius, -1.f, 1.f));
    float v = (theta - thetaMin) / (thetaMax - thetaMin);
    // compute HSphere dp/du and dp/dv
    float zradius = sqrtf(phit.x * phit.x + phit.y * phit.y);
    float invradius = 1.f / zradius;
    float cosphi = phit.x * invradius;
    float sinphi = phit.y * invradius;
    Vector3 dpdu(-phiMax * phit.y, phiMax * phit.x, 0);
    Vector3 dpdv = (thetaMax - thetaMin) *
      Vector3(phit.z * cosphi, phit.z * sinphi, -radius * sinf(theta));
    // compute HSphere dn/du and dn/dv
    // TODO pg 121
    Normal dndu, dndv;
    // initialize diff. geom. from parametric information
    const Transform &o2w = *objectToWorld;
    *dg = DifferentialGeometry(o2w(phit), o2w(dpdu), o2w(dpdv),
                               o2w(dndu), o2w(dndv), u , v, this);
    // update tHit for quadric intersection
    *tHit = thit;
    // compute HRayEpsilon for quadric intersection
    *HRayEpsilon = 5e-4f * *tHit;
    return true;
  }

  bool HSphere::intersectP(const HRay &r) const {
    // float phi;
    // Point3 phit;
    // transform HRay to object space
    // compute quadratic HSphere coefficients
    // solve quadratic equation for t values
    // compute HSphere hit position and  phi
    // test HSphere intersection against clipping parameters
    return true;
  }

  float HSphere::surfaceArea() const {
    return phiMax * radius * (zmax - zmin);
  }

  } // helios namespace
