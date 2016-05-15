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
} // ponos namespace
