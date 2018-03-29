/*
 * Copyright (c) 2017 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
*/

#ifndef PONOS_GEOMETRY_BBOX_H
#define PONOS_GEOMETRY_BBOX_H

#include <ponos/geometry/point.h>

#include <algorithm>
#include <iostream>

namespace ponos {

class BBox2D {
public:
  BBox2D();
  BBox2D(const Point2 &p1, const Point2 &p2);

  static BBox2D unitBox() { return {Point2(), Point2(1, 1)}; }

  bool inside(const Point2 &p) const {
    return (p.x >= pMin.x && p.x <= pMax.x && p.y >= pMin.y && p.y <= pMax.y);
  }

  float size(int d) const {
    d = std::max(0, std::min(1, d));
    return pMax[d] - pMin[d];
  }

  Vector2 extends() const { return pMax - pMin; }

  Point2 center() const { return pMin + (pMax - pMin) * .5f; }

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
  explicit BBox(const Point3 &p);
  /// Creates a bounding box of 2r side centered at c
  /// \param c center point
  /// \param r radius
  BBox(const Point3& c, float r);
  BBox(const Point3 &p1, const Point3 &p2);
  friend BBox make_union(const BBox &b, const BBox &b1);
  static BBox unitBox() { return {Point3(), Point3(1, 1, 1)}; }
  bool overlaps(const BBox &b) const {
    bool x = (pMax.x >= b.pMin.x) && (pMin.x <= b.pMax.x);
    bool y = (pMax.y >= b.pMin.y) && (pMin.y <= b.pMax.y);
    bool z = (pMax.z >= b.pMin.z) && (pMin.z <= b.pMax.z);
    return (x && y && z);
  }
  bool contains(const Point3 &p) const {
    return (p.x >= pMin.x && p.x <= pMax.x && p.y >= pMin.y && p.y <= pMax.y &&
        p.z >= pMin.z && p.z <= pMax.z);
  }
  /// \param b bbox
  /// \return true if bbox is fully inside
  bool contains(const BBox &b) const {
    return contains(b.pMin) && contains(b.pMax);
  }
  void expand(float delta) {
    pMin -= Vector3(delta, delta, delta);
    pMax -= Vector3(delta, delta, delta);
  }
  /**
 * y
 * |_ x
 * z
 *   /  2  /  3 /
 *  / 6  /  7  /
 *  ------------
 * |   0 |   1 |
 * | 4   | 5   |
 * ------------
 */
  std::vector<BBox> splitBy8() const {
    auto mid = center();
    std::vector<BBox> children;
    children.emplace_back(pMin, mid);
    children.emplace_back(Point3(mid.x, pMin.y, pMin.z), Point3(pMax.x, mid.y, mid.z));
    children.emplace_back(Point3(pMin.x, mid.y, pMin.z), Point3(mid.x, pMax.y, mid.z));
    children.emplace_back(Point3(mid.x, mid.y, pMin.z), Point3(pMax.x, pMax.y, mid.z));
    children.emplace_back(Point3(pMin.x, pMin.y, mid.z), Point3(mid.x, mid.y, pMax.z));
    children.emplace_back(Point3(mid.x, pMin.y, mid.z), Point3(pMax.x, mid.y, pMax.z));
    children.emplace_back(Point3(pMin.x, mid.y, mid.z), Point3(mid.x, pMax.y, pMax.z));
    children.emplace_back(Point3(mid.x, mid.y, mid.z), Point3(pMax.x, pMax.y, pMax.z));
    return children;
  }
  Point3 center() const { return pMin + (pMax - pMin) * .5f; }
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
  BBox2D xy() const { return BBox2D(pMin.xy(), pMax.xy()); }
  BBox2D yz() const { return BBox2D(pMin.yz(), pMax.yz()); }
  BBox2D xz() const { return BBox2D(pMin.xz(), pMax.xz()); }
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
