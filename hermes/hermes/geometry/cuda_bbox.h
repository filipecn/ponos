/*
 * Copyright (c) 2019 FilipeCN
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

#ifndef HERMES_GEOMETRY_CUDA_BBOX_H
#define HERMES_GEOMETRY_CUDA_BBOX_H

#include <hermes/geometry/cuda_point.h>

#include <algorithm>
#include <iostream>

namespace hermes {

namespace cuda {

template <typename T> class BBox2 {
public:
  __host__ __device__ BBox2();
  __host__ __device__ explicit BBox2(const Point2<T> &p);
  __host__ __device__ BBox2(const Point2<T> &p1, const Point2<T> &p2);
  __host__ __device__ static BBox2 unitBox();
  __host__ __device__ bool contains(const Point2<T> &p) const;
  __host__ __device__ T size(int d) const;
  __host__ __device__ Vector2<T> extends() const;
  __host__ __device__ Point2<T> center() const;
  __host__ __device__ Point2<T> centroid() const;
  __host__ __device__ int maxExtent() const;
  __host__ __device__ const Point2<T> &operator[](int i) const;
  __host__ __device__ Point2<T> &operator[](int i);

  Point2<T> lower, upper;
};

typedef BBox2<float> bbox2;

template <typename T>
__host__ __device__ inline BBox2<T> make_union(const BBox2<T> &b,
                                               const Point2<T> &p) {
  BBox2<T> ret = b;
  ret.lower.x = std::min(b.lower.x, p.x);
  ret.lower.y = std::min(b.lower.y, p.y);
  ret.upper.x = std::max(b.upper.x, p.x);
  ret.upper.y = std::max(b.upper.y, p.y);
  return ret;
}

template <typename T>
__host__ __device__ inline BBox2<T> make_union(const BBox2<T> &a,
                                               const BBox2<T> &b) {
  BBox2<T> ret = make_union(a, b.lower);
  return make_union(ret, b.upper);
}

/// Axis-aligned region of space.
/// \tparam T coordinates type
template <typename T> class BBox3 {
public:
  /// Creates an empty bounding box
  __host__ __device__ BBox3();
  /// Creates a bounding enclosing a single point
  /// \param p point
  __host__ __device__ explicit BBox3(const Point3<T> &p);
  /// Creates a bounding box of 2r side centered at c
  /// \param c center point
  /// \param r radius
  __host__ __device__ BBox3(const Point3<T> &c, T r);
  /// Creates a bounding box enclosing two points
  /// \param p1 first point
  /// \param p2 second point
  __host__ __device__ BBox3(const Point3<T> &p1, const Point3<T> &p2);
  /// \param p
  /// \return true if this bounding box encloses **p**
  __host__ __device__ bool contains(const Point3<T> &p) const;
  /// \param b bbox
  /// \return true if bbox is fully inside
  __host__ __device__ bool contains(const BBox3 &b) const;
  /// Doesn't consider points on the upper boundary to be inside the bbox
  /// \param p point
  /// \return true if contains exclusive
  __host__ __device__ bool containsExclusive(const Point3<T> &p) const;
  /// Pads the bbox in both dimensions
  /// \param delta expansion factor (lower - delta, upper + delta)
  __host__ __device__ void expand(T delta);
  /// \return vector along the diagonal upper - lower
  __host__ __device__ Vector3<T> diagonal() const;
  /// \return index of longest axis
  __host__ __device__ int maxExtent() const;
  /// \param i 0 = lower, 1 = upper
  /// \return lower or upper point
  __host__ __device__ const Point3<T> &operator[](int i) const;
  /// \param i 0 = lower, 1 = upper
  /// \return lower or upper point
  __host__ __device__ Point3<T> &operator[](int i);
  /// \param c corner index
  /// \return corner point
  __host__ __device__ Point3<T> corner(int c) const;
  /// \param p point
  /// \return position of **p** relative to the corners where lower has offset
  /// (0,0,0) and upper (1,1,1)
  __host__ __device__ Vector3<T> offset(const Point3<T> &p) const;
  /// \return surface area of the six faces
  __host__ __device__ T surfaceArea() const;
  /// \return volume inside the bounds
  __host__ __device__ T volume() const;
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
  //   std::vector<BBox3> splitBy8() const;
  __host__ __device__ Point3<T> center() const;
  __host__ __device__ T size(size_t d) const;
  __host__ __device__ BBox2<T> xy() const;
  __host__ __device__ BBox2<T> yz() const;
  __host__ __device__ BBox2<T> xz() const;
  __host__ __device__ Point3<T> centroid() const;
  template <typename TT>
  __host__ __device__ friend BBox3 make_union(const BBox3 &b,
                                              const BBox3<T> &b1);
  __host__ __device__ static BBox3 unitBox();

  Point3<T> lower, upper;
};

using bbox2 = BBox2<float>;
using bbox2f = BBox2<float>;
using bbox3 = BBox3<float>;
using bbox3f = BBox3<float>;
/// Checks if both bounding boxes overlap
/// \param a first bounding box
/// \param b second bounding box
/// \return true if they overlap
template <typename T>
__host__ __device__ bool overlaps(const BBox3<T> &a, const BBox3<T> &b);
/// \tparam T coordinates type
/// \param b bounding box
/// \param p point
/// \return a new bounding box that encopasses **b** and **p**
template <typename T>
__host__ __device__ BBox3<T> make_union(const BBox3<T> &b, const Point3<T> &p);

inline __host__ __device__ bbox3 make_union(const bbox3 &a, const bbox3 &b) {
  bbox3 ret = make_union(a, b.lower);
  return make_union(ret, b.upper);
}
/// \tparam T coordinates type
/// \param a bounding box
/// \param b bounding box
/// \return a new bounding box that encopasses **a** and **b**
template <typename T>
__host__ __device__ BBox3<T> make_union(const BBox3<T> &a, const BBox3<T> &b);
/// \tparam T coordinates type
/// \param a bounding box
/// \param b bounding box
/// \return a new bbox resulting from the intersection of **a** and **b**
template <typename T>
__host__ __device__ BBox3<T> intersect(const BBox3<T> &a, const BBox3<T> &b);

#include "cuda_bbox.inl"

} // namespace cuda

} // namespace hermes

#endif
