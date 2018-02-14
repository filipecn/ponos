/*
 * Copyright (c) 2018 FilipeCN
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

#ifndef PONOS_Z_POINT_SET_H
#define PONOS_Z_POINT_SET_H

#include <ponos/structures/point_set_interface.h>
#include <ponos/geometry/transform.h>
#include <ponos/structures/octree.h>
#include <ponos/algorithm/search.h>

namespace ponos {

/*! Arranges a set of points in z-order.
 * Provides neighborhood queries via
 * k-nn, radial search and closest point.
 * Allows dynamic addition and removal of points.
 * TODO: put complexity costs
 */
class ZPointSet : public PointSetInterface {
public:
  friend class iterator;
  struct PointElement {
    PointElement() : id(0), zcode(0) { active = true; }
    uint id;    ///< element id for reference from outside code
    uint zcode; ///< computed morton code
    bool active;  ///< element existence
  };
  class iterator {
  public:
    /// \param s z point set reference
    /// \param f first element id (morton code)
    /// \param depth tree level
    iterator(ZPointSet &s, uint f = 0, uint depth = 0) :
        first_(f), cur_(0), last_(0), zps_(s) {
      comp_ = [](const PointElement &p, const uint &v) {
        if (p.zcode < v)
          return -1;
        if (p.zcode > v)
          return 1;
        return 0;
      };
      if (zps_.end_ == 0)
        return;
      int firstIndex = lower_bound<PointElement, size_t>(&zps_.points_[0], zps_.end_,
                                                 first_, comp_);
      last_ = zps_.end_;
      if (depth) {
        last_ = first_ + (1 << ((zps_.nbits_ - depth) * 2));
        last_ = lower_bound<PointElement, size_t>(&zps_.points_[0], zps_.end_,
                                                 last_, comp_) + 1;
      }
      first_ = firstIndex + 1;
      cur_ = first_;
    }
    /// \return true if there is more elements to iterate
    bool next() const { return cur_ < last_; }
    /// \return position of the current element in world coordinates
    Point3 getWorldPosition() {
      FATAL_ASSERT(cur_ < last_);
      return zps_.positions_[zps_.points_[cur_].id];
    }
    /// sets a new position to current element
    /// \param p new position value
    void setPosition(const Point3 &p) {
      zps_.positions_[zps_.points_[cur_].id] = p;
      Point3 gp = zps_.toSet_(p);
      zps_.points_[cur_].zcode = computeIndex(gp);
    }
    /// \return id of current element
    uint getId() {
      FATAL_ASSERT(cur_ < last_);
      return zps_.points_[cur_].id;
    }
    /// \return pointer to current point element struct
    PointElement *pointElement() { return &zps_.points_[cur_]; }
    /// advance on iteration
    void operator++() { cur_++; }
    /// \return number of elements to iterate (first -> last)
    int count() const { return last_ - first_; }
  private:
    uint first_, cur_, last_;
    ZPointSet &zps_;
    std::function<int(const PointElement &p, const uint &v)> comp_;
  };
  /// Default constructor
  ZPointSet();
  /// Default destructor
  ~ZPointSet();
  /// Constructor
  /// \param maxCoordinates maximum coordinates value for an input point
  ZPointSet(uint maxCoordinates);
  /// necessary before any search after position modification
  void update();
  // INTERFACE
  /// active points count
  /// \return number of active points
  uint size() override;
  /// adds new point with position **p**
  /// \param p position to be added
  /// \return element id, so this position can be accessed later
  uint add(Point3 p) override;
  /// sets position **p** to element **i**
  /// \param i element id
  /// \param p new position value
  void setPosition(unsigned int i, Point3 p) override;
  /// removes element **i**
  /// \param i element id
  void remove(uint i) override;
  /// random access operator
  /// \param i point index
  /// \return position of point **i**
  Point3 operator[](uint i) const override;
  //void search(const BBox &b,
  //            const std::function<void(uint)> &f) override;
  //void
  //iteratePoints(const std::function<void(size_t, Point3)> &f) const override;
  // METHODS
  //void update();
private:
  /// morton code transform
  /// \return morton code of coordinate **p**
  static uint computeIndex(const Point3 &p);
  Point3 resolution_;   ///< maximum coordinates
  Transform toSet_;     ///< map to underling grid
  std::vector<PointElement> points_; ///< array of point elements
  std::vector<Point3> positions_;    ///< points positions
  std::vector<uint> indices_;        ///< map _points -> _positions
  uint end_;          ///< current number of active point element
  uint lastId_;       ///< last point element id generated
  uint nbits_;        ///< number of bits used by the maximum coordinate value
  uint maxDepth_;     ///< maximum depth of search tree, bounded by nbits
  uint maxZCode_;     ///< maximum allowed morton code
  bool needUpdate_;   ///< indicates if elements must be sorted
  bool needZUpdate_;  ///< indicates if morton codes need to be recalculated
  bool sizeChanged_;  ///< indicates if points have been added/removed
};

} // ponos namespace

#endif //PONOS_Z_POINT_SET_H
