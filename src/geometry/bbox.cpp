#include "geometry/bbox.h"

namespace ponos {

  BBox::BBox() {
    pMin = Point3( INFINITY, INFINITY, INFINITY);
    pMax = Point3(-INFINITY,-INFINITY,-INFINITY);
  }

  BBox::BBox(const Point3& p)
    : pMin(p), pMax(p) {}

  BBox::BBox(const Point3& p1, const Point3& p2) {
    pMin = Point3(std::min(p1.x, p2.x), std::min(p1.y, p2.y), std::min(p1.z, p2.z));
    pMax = Point3(std::max(p1.x, p2.x), std::max(p1.y, p2.y), std::max(p1.z, p2.z));
  }

};
