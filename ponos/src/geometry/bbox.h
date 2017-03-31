#ifndef PONOS_GEOMETRY_BBOX_H
#define PONOS_GEOMETRY_BBOX_H

#include "geometry/point.h"
#include "geometry/vector.h"

#include <iostream>
#include <algorithm>

namespace ponos {

class BBox2D {
public:
  BBox2D();
  BBox2D(const Point2 &p1, const Point2 &p2);

  bool inside(const Point2 &p) const {
    return (p.x >= pMin.x && p.x <= pMax.x && p.y >= pMin.y && p.y <= pMax.y);
  }

  float size(int d) const {
    d = std::max(0, std::min(1, d));
    return pMax[d] - pMin[d];
  }

  Point2 center() { return pMin + (pMax - pMin) * .5f; }

  Point2 centroid() const { return pMin * .5f + vec2(pMax * .5f); }

  int maxExtent() const {
    Vector2 diag = pMax - pMin;
    if (diag.x > diag.y)
      return 0;
    return 1;
  }

  const Point2 &operator[](int i) const { return (i == 0) ? pMin : pMax; }
  Point2 &operator[](int i) { return (i == 0) ? pMin : pMax; }

  Point2 pMin, pMax;
};

inline BBox2D make_union(const BBox2D &b, const Point2 &p) {
  BBox2D ret = b;
  ret.pMin.x = std::min(b.pMin.x, p.x);
  ret.pMin.y = std::min(b.pMin.y, p.y);
  ret.pMax.x = std::max(b.pMax.x, p.x);
  ret.pMax.y = std::max(b.pMax.y, p.y);
  return ret;
}

inline BBox2D make_union(const BBox2D &a, const BBox2D &b) {
  BBox2D ret = make_union(a, b.pMin);
  return make_union(ret, b.pMax);
}

class BBox {
public:
  BBox();
  BBox(const Point3 &p);
  BBox(const Point3 &p1, const Point3 &p2);
  friend BBox make_union(const BBox &b, const BBox &b1);
  bool overlaps(const BBox &b) const {
    bool x = (pMax.x >= b.pMin.x) && (pMin.x <= b.pMax.x);
    bool y = (pMax.y >= b.pMin.y) && (pMin.y <= b.pMax.y);
    bool z = (pMax.z >= b.pMin.z) && (pMin.z <= b.pMax.z);
    return (x && y && z);
  }
  bool inside(const Point3 &p) const {
    return (p.x >= pMin.x && p.x <= pMax.x && p.y >= pMin.y && p.y <= pMax.y &&
            p.z >= pMin.z && p.z <= pMax.z);
  }
  void expand(float delta) {
    pMin -= Vector3(delta, delta, delta);
    pMax -= Vector3(delta, delta, delta);
  }
  float size(size_t d) const { return pMax[d] - pMin[d]; }
  float surfaceArea() const {
    Vector3 d = pMax - pMin;
    return 2.f * (d.x * d.y + d.x * d.z + d.y * d.z);
  }
  float volume() const {
    Vector3 d = pMax - pMin;
    return d.x * d.y * d.z;
  }
  int maxExtent() const {
    Vector3 diag = pMax - pMin;
    if (diag.x > diag.y && diag.x > diag.z)
      return 0;
    else if (diag.y > diag.z)
      return 1;
    return 2;
  }
  Point3 centroid() const { return pMin * .5f + vec3(pMax * .5f); }
  const Point3 &operator[](int i) const { return (i == 0) ? pMin : pMax; }
  Point3 &operator[](int i) { return (i == 0) ? pMin : pMax; }

  Point3 pMin, pMax;
};

inline BBox make_union(const BBox &b, const Point3 &p) {
  BBox ret = b;
  ret.pMin.x = std::min(b.pMin.x, p.x);
  ret.pMin.y = std::min(b.pMin.y, p.y);
  ret.pMin.z = std::min(b.pMin.z, p.z);
  ret.pMax.x = std::max(b.pMax.x, p.x);
  ret.pMax.y = std::max(b.pMax.y, p.y);
  ret.pMax.z = std::max(b.pMax.z, p.z);
  return ret;
}

inline BBox make_union(const BBox &a, const BBox &b) {
  BBox ret = make_union(a, b.pMin);
  return make_union(ret, b.pMax);
}

} // ponos namespace

#endif
