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

template<typename T> class BBox1 {
public:
  BBox1() {
    lower = Constants::greatest<T>();
    upper = Constants::lowest<T>();
  }
  explicit BBox1(const T &p) : lower(p), upper(p) {}
  BBox1(const T &p1, const T &p2) : lower(std::min(p1, p2)),
                                    upper(std::max(p1, p2)) {}
  static BBox1 unitBox() { return BBox1<T>(0, 1); }
  bool contains(const T &p) const { return p >= lower && p <= upper; }
  T extends() const { return upper - lower; }
  T center() const { return lower + (upper - lower) * 0.5; }
  T centroid() const { return lower * .5 + upper * .5; }
  const T &operator[](int i) const { return (&lower)[i]; }
  T &operator[](int i) { return (&lower)[i]; }

  T lower, upper;
};

typedef BBox1<real_t> bbox1;

template<typename T>
BBox1<T> make_union(const BBox1<T> &b, const T &p) {
  BBox1 <T> ret = b;
  ret.lower = std::min(b.lower, p);
  ret.upper = std::max(b.upper, p);
  return ret;
}

template<typename T>
BBox1<T> make_union(const BBox1<T> &a, const BBox1<T> &b) {
  BBox1<T> ret = make_union(a, b.lower);
  return make_union(ret, b.upper);
}
template<typename T> class BBox2 {
public:
  BBox2() {
    lower = Point2<T>(Constants::greatest<T>());
    upper = Point2<T>(Constants::lowest<T>());
  }
  explicit BBox2(const Point2<T> &p) : lower(p), upper(p) {}
  BBox2(const Point2<T> &p1, const Point2<T> &p2) {
    lower = Point2<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y));
    upper = Point2<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y));
  }
  static BBox2<T> unitBox() {
    return {Point2<T>(), Point2<T>(1, 1)};
  }
  bool contains(const Point2<T> &p) const {
    return (p.x >= lower.x && p.x <= upper.x && p.y >= lower.y
        && p.y <= upper.y);
  }
  real_t size(int d) const {
    d = std::max(0, std::min(1, d));
    return upper[d] - lower[d];
  }
  Vector2<T> extends() const {
    return upper - lower;
  }
  Point2<T> center() const {
    return lower + (upper - lower) * .5f;
  }
  Point2<T> centroid() const {
    return lower * .5f + vec2(upper * .5f);
  }
  int maxExtent() const {
    Vector2<T> diag = upper - lower;
    if (diag.x > diag.y)
      return 0;
    return 1;
  }
  const Point2<T> &operator[](int i) const {
    return (i == 0) ? lower : upper;
  }
  Point2<T> &operator[](int i) {
    return (i == 0) ? lower : upper;
  }

  Point2<T> lower, upper;
};

typedef BBox2<real_t> bbox2;

template<typename T>
inline BBox2<T> make_union(const BBox2<T> &b, const Point2<T> &p) {
  BBox2<T> ret = b;
  ret.lower.x = std::min(b.lower.x, p.x);
  ret.lower.y = std::min(b.lower.y, p.y);
  ret.upper.x = std::max(b.upper.x, p.x);
  ret.upper.y = std::max(b.upper.y, p.y);
  return ret;
}

template<typename T>
inline BBox2<T> make_union(const BBox2<T> &a, const BBox2<T> &b) {
  BBox2<T> ret = make_union(a, b.lower);
  return make_union(ret, b.upper);
}

/// Axis-aligned region of space.
/// \tparam T coordinates type
template<typename T> class BBox3 {
public:
  /// Creates an empty bounding box
  BBox3() {
    lower = Point3<T>(Constants::greatest<T>());
    upper = Point3<T>(Constants::lowest<T>());
  }
  /// Creates a bounding enclosing a single point
  /// \param p point
  explicit BBox3(const Point3<T> &p) : lower(p), upper(p) {}
  /// Creates a bounding box of 2r side centered at c
  /// \param c center point
  /// \param r radius
  BBox3(const Point3<T> &c, real_t r) {
    lower = c - Vector3<T>(r, r, r);
    upper = c + Vector3<T>(r, r, r);
  }
  /// Creates a bounding box enclosing two points
  /// \param p1 first point
  /// \param p2 second point
  BBox3(const Point3<T> &p1, const Point3<T> &p2) {
    lower = Point3<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y),
                      std::min(p1.z, p2.z));
    upper = Point3<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y),
                      std::max(p1.z, p2.z));
  }
  /// \param p
  /// \return true if this bounding box encloses **p**
  bool contains(const Point3<T> &p) const {
    return (p.x >= lower.x && p.x <= upper.x && p.y >= lower.y &&
        p.y <= upper.y && p.z >= lower.z && p.z <= upper.z);
  }
  /// \param b bbox
  /// \return true if bbox is fully inside
  bool contains(const BBox3 &b) const {
    return contains(b.lower) && contains(b.upper);
  }
  /// Doesn't consider points on the upper boundary to be inside the bbox
  /// \param p point
  /// \return true if contains exclusive
  bool containsExclusive(const Point3<T> &p) const {
    return (p.x >= lower.x && p.x < upper.x && p.y >= lower.y && p.y < upper.y
        && p.z >= lower.z && p.z < upper.z);
  }
  /// Pads the bbox in both dimensions
  /// \param delta expansion factor (lower - delta, upper + delta)
  void expand(real_t delta) {
    lower -= Vector3<T>(delta, delta, delta);
    upper += Vector3<T>(delta, delta, delta);
  }
  /// \return vector along the diagonal upper - lower
  Vector3<T> diagonal() const {
    return upper - lower;
  }
  /// \return index of longest axis
  int maxExtent() const {
    Vector3<T> diag = upper - lower;
    if (diag.x > diag.y && diag.x > diag.z)
      return 0;
    else if (diag.y > diag.z)
      return 1;
    return 2;
  }
  /// \param i 0 = lower, 1 = upper
  /// \return lower or upper point
  const Point3<T> &operator[](int i) const {
    return (i == 0) ? lower : upper;
  }
  /// \param i 0 = lower, 1 = upper
  /// \return lower or upper point
  Point3<T> &operator[](int i) {
    return (i == 0) ? lower : upper;
  }
  /// \param c corner index
  /// \return corner point
  Point3<T> corner(int c) const {
    return Point3<T>((*this)[(c & 1)].x, (*this)[(c & 2) ? 1 : 0].y,
                     (*this)[(c & 4) ? 1 : 0].z);
  }
  /// \param p point
  /// \return position of **p** relative to the corners where lower has offset
  /// (0,0,0) and upper (1,1,1)
  Vector3<T> offset(const Point3<T> &p) const {
    ponos::Vector3<T> o = p - lower;
    if (upper.x > lower.x)
      o.x /= upper.x - lower.x;
    if (upper.y > lower.y)
      o.y /= upper.y - lower.y;
    if (upper.z > lower.z)
      o.z /= upper.z - lower.z;
    return o;
  }
  /// \return surface area of the six faces
  T surfaceArea() const {
    Vector3<T> d = upper - lower;
    return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
  }
  /// \return volume inside the bounds
  T volume() const {
    Vector3<T> d = upper - lower;
    return d.x * d.y * d.z;
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
  std::vector<BBox3> splitBy8() const {
    auto mid = center();
    std::vector<BBox3<T>> children;
    children.emplace_back(lower, mid);
    children.emplace_back(Point3<T>(mid.x, lower.y, lower.z),
                          Point3<T>(upper.x, mid.y, mid.z));
    children.emplace_back(Point3<T>(lower.x, mid.y, lower.z),
                          Point3<T>(mid.x, upper.y, mid.z));
    children.emplace_back(Point3<T>(mid.x, mid.y, lower.z),
                          Point3<T>(upper.x, upper.y, mid.z));
    children.emplace_back(Point3<T>(lower.x, lower.y, mid.z),
                          Point3<T>(mid.x, mid.y, upper.z));
    children.emplace_back(Point3<T>(mid.x, lower.y, mid.z),
                          Point3<T>(upper.x, mid.y, upper.z));
    children.emplace_back(Point3<T>(lower.x, mid.y, mid.z),
                          Point3<T>(mid.x, upper.y, upper.z));
    children.emplace_back(Point3<T>(mid.x, mid.y, mid.z),
                          Point3<T>(upper.x, upper.y, upper.z));
    return children;
  }
  Point3<T> center() const {
    return lower + (upper - lower) * .5f;
  }
  T size(u32 d) const {
    return upper[d] - lower[d];
  }
  BBox2<T> xy() const {
    return BBox2<T>(lower.xy(), upper.xy());
  }
  BBox2<T> yz() const {
    return BBox2<T>(lower.yz(), upper.yz());
  }
  BBox2<T> xz() const {
    return BBox2<T>(lower.xz(), upper.xz());
  }
  Point3<T> centroid() const {
    return lower * .5f + vec3(upper * .5f);
  }
  template<typename TT>
  friend BBox3 make_union(const BBox3 &b, const BBox3<T> &b1);
  static BBox3 unitBox() {
    return {Point3<T>(), Point3<T>(1, 1, 1)};
  }

  Point3<T> lower, upper;
};

typedef BBox3<real_t> bbox3;
typedef BBox3<float> bbox3f;

/// Checks if both bounding boxes overlap
/// \param a first bounding box
/// \param b second bounding box
/// \return true if they overlap
template<typename T> bool overlaps(const BBox3<T> &a, const BBox3<T> &b) {
  bool x = (a.upper.x >= b.lower.x) && (a.lower.x <= b.upper.x);
  bool y = (a.upper.y >= b.lower.y) && (a.lower.y <= b.upper.y);
  bool z = (a.upper.z >= b.lower.z) && (a.lower.z <= b.upper.z);
  return (x && y && z);
}
/// \tparam T coordinates type
/// \param b bounding box
/// \param p point
/// \return a new bounding box that encopasses **b** and **p**
template<typename T>
BBox3<T> make_union(const BBox3<T> &b, const Point3<T> &p) {
  BBox3<T> ret = b;
  ret.lower.x = std::min(b.lower.x, p.x);
  ret.lower.y = std::min(b.lower.y, p.y);
  ret.lower.z = std::min(b.lower.z, p.z);
  ret.upper.x = std::max(b.upper.x, p.x);
  ret.upper.y = std::max(b.upper.y, p.y);
  ret.upper.z = std::max(b.upper.z, p.z);
  return ret;
}

inline bbox3 make_union(const bbox3 &a, const bbox3 &b) {
  bbox3 ret = make_union(a, b.lower);
  return make_union(ret, b.upper);
}
/// \tparam T coordinates type
/// \param a bounding box
/// \param b bounding box
/// \return a new bounding box that encopasses **a** and **b**
template<typename T> BBox3<T> make_union(const BBox3<T> &a, const BBox3<T> &b) {
  BBox3<T> ret = make_union(a, b.lower);
  return make_union(ret, b.upper);
}
/// \tparam T coordinates type
/// \param a bounding box
/// \param b bounding box
/// \return a new bbox resulting from the intersection of **a** and **b**
template<typename T> BBox3<T> intersect(const BBox3<T> &a, const BBox3<T> &b) {
  return BBox3<T>(
      Point3<T>(std::max(a.lower.x, b.lower.x), std::max(a.lower.x, b.lower.y),
                std::max(a.lower.z, b.lower.z)),
      Point3<T>(std::min(a.lower.x, b.lower.x), std::min(a.lower.x, b.lower.y),
                std::min(a.lower.z, b.lower.z)));
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const BBox1<T> &b) {
  os << "BBox1(" << b.lower << ", " << b.upper << ")";
  return os;
}
template<typename T>
std::ostream &operator<<(std::ostream &os, const BBox2<T> &b) {
  os << "BBox2(" << b.lower << ", " << b.upper << ")";
  return os;
}
template<typename T>
std::ostream &operator<<(std::ostream &os, const BBox3<T> &b) {
  os << "BBox3(" << b.lower << ", " << b.upper << ")";
  return os;
}

} // namespace ponos

#endif
