#include "shapes/sphere.h"

#include "geometry/utils.h"

namespace ponos {

  Sphere::Sphere(const Transform *o2w, const Transform *w2o, bool ro,
                 float rad, float z0, float z1, float pm)
    : Shape(o2w, w2o, ro) {
      zmin = clamp(std::min(z0, z1), -radius, radius);
    radius = rad;
    zmax = clamp(std::max(z0, z1), -radius, radius);
    thetaMin = acosf(clamp(zmin / radius, -1.f, 1.f));
    thetaMax = acosf(clamp(zmin / radius, -1.f, 1.f));
    thetaMax = TO_RADIANS(clamp(pm, 0.f, 360.f));
  }

  BBox Sphere::objectBound() const {
    return BBox(Point3(-radius, -radius, zmin), Point3(radius, radius, zmax));
  }

  bool Sphere::intersect(const Ray &r, float *tHit, float *rayEpsilon,
                         DifferentialGeometry *dg) const {
    float phi;
    Point3 phit;
    // transform ray to object space
    Ray ray;
    (*worldToObject)(r, &ray);
    // compute quadritic sphere coefficients
    float A = ray.d.x * ray.d.x + ray.d.y * ray.d.y + ray.d.z * ray.d.z;
    float B = 2 * (ray.d.x * ray.o.x + ray.d.y * ray.o.y + ray.d.z * ray.o.z);
    float C = ray.o.x * ray.o.x + ray.o.y * ray.o.y + ray.o.z * ray.o.z - radius * radius;
    // solve quadritic equation for t values
    float t0, t1;
    if(!solve_quadratic(A, B, C, t0, t1))
      return false;
    if(t0 > ray.max_t || t1 < ray.min_t)
      return false;
    float thit = t0;
    if(t0 < ray.min_t) {
      thit = t1;
      if(thit > ray.max_t)
        return false;
    }
    // compute sphere hit position and phi
    phit = ray(thit);
    if(IS_ZERO(phit.x) && IS_ZERO(phit.y))
      phit.x = 1e-5f * radius;
    phi = atan2f(phit.y, phit.x);
    if(phi < 0.) phi += 2.f * PI;
    // test sphere intersection against clipping parameters
    if((zmin > -radius && phit.z < zmin) ||
       (zmax <  radius && phit.z > zmax) || phi > phiMax) {
      if(thit == t1) return false;
      if(t1 > ray.max_t) return false;
      thit = t1;
      // compute sphere hit position and phi
      phit = ray(thit);
      if(IS_ZERO(phit.x) && IS_ZERO(phit.y))
        phit.x = 1e-5f * radius;
      phi = atan2f(phit.y, phit.x);
      if(phi < 0.) phi += 2.f * PI;
      if((zmin > -radius && phit.z < zmin) ||
         (zmax <  radius && phit.z > zmax) || phi > phiMax)
          return false;
    }
    // find parametric representation of sphere hit
    float u = phi / phiMax;
    float theta = acosf(clamp(phit.z / radius, -1.f, 1.f));
    float v = (theta - thetaMin) / (thetaMax - thetaMin);
    // compute sphere dp/du and dp/dv
    float zradius = sqrtf(phit.x * phit.x + phit.y * phit.y);
    float invradius = 1.f / zradius;
    float cosphi = phit.x * invradius;
    float sinphi = phit.y * invradius;
    Vector3 dpdu(-phiMax * phit.y, phiMax * phit.x, 0);
    Vector3 dpdv = (thetaMax - thetaMin) *
      Vector3(phit.z * cosphi, phit.z * sinphi, -radius * sinf(theta));
    // compute sphere dn/du and dn/dv
    // TODO pg 121
    Normal dndu, dndv;
    // initialize diff. geom. from parametric information
    const Transform &o2w = *objectToWorld;
    *dg = DifferentialGeometry(o2w(phit), o2w(dpdu), o2w(dpdv),
                               o2w(dndu), o2w(dndv), u , v, this);
    // update tHit for quadric intersection
    *tHit = thit;
    // compute rayEpsilon for quadric intersection
    *rayEpsilon = 5e-4f * *tHit;
    return true;
  }

  bool Sphere::intersectP(const Ray &r) const {
    float phi;
    Point3 phit;
    // transform ray to object space
    // compute quadratic sphere coefficients
    // solve quadratic equation for t values
    // compute sphere hit position and  phi
    // test sphere intersection against clipping parameters
    return true;
  }

  float Sphere::surfaceArea() const {
    return phiMax * radius * (zmax - zmin);
  }

  } // ponos namespace
