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
#include <ponos/geometry/queries.h>

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
  /// Helper class for iterating elements
  class iterator {
  public:
    /// \param s z point set reference
    /// \param f first element id (morton code)
    /// \param depth tree level
    explicit iterator(ZPointSet &s, uint f = 0, uint depth = 0) :
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
      int firstIndex = lower_bound<PointElement, uint>(&zps_.points_[0], zps_.end_,
                                                       first_, comp_);
      last_ = zps_.end_;
      FATAL_ASSERT(depth <= zps_.maxDepth_);
      if (depth) {
        last_ = first_ + (1 << ((zps_.nbits_ - depth) * 3));
        int lb =
            lower_bound<PointElement, uint>(&zps_.points_[std::max(firstIndex, 0)], zps_.end_ - std::max(firstIndex, 0),
                                            last_, comp_) + 1 + std::max(firstIndex, 0);
        last_ = static_cast<uint>(lb);
      }
      first_ = static_cast<uint>(firstIndex + 1);
      cur_ = first_;
      FATAL_ASSERT(last_ <= zps_.end_);
      FATAL_ASSERT(first_ <= last_);
    }
    /// \return true if there is more elements to iterate
    bool next() const { return cur_ < last_; }
    /// \return position of the current element in world coordinates
    Point3 getWorldPosition() {
      FATAL_ASSERT(cur_ < last_);
      FATAL_ASSERT(cur_ < zps_.points_.size());
      FATAL_ASSERT(zps_.points_[cur_].id < zps_.positions_.size());
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
    uint count() const { return last_ - first_; }
  private:
    uint first_, cur_, last_;
    ZPointSet &zps_;
    std::function<int(const PointElement &p, const uint &v)> comp_;
  };
  /// Helper class for search queries
  class search_tree : public Octree<uint> {
  public:
    /// \param z z point set reference
    /// \param f **[optional]** refinement criteria
    explicit search_tree(ZPointSet &z, std::function<bool(uint, uint)> f =
    [](uint id, uint depth) -> bool {
      UNUSED_VARIABLE(id);
      UNUSED_VARIABLE(depth);
      return true;
    })
        : Octree<uint>(), zps_(z) {
      this->root_ = new Octree<uint>::Node(BBox(Point3(), zps_.resolution_));
      this->count_++;
      this->root_->data = 0;
      this->refine(this->root_,
                   [&](Octree<uint>::Node &node) -> bool {
                     if (node.level() >= zps_.maxDepth_)
                       return false;
                     return f(node.data, node.level());
                   },
                   [&](Octree<uint>::Node &node) {
                     uint d = (zps_.nbits_ - node.level() - 1) * 3;
                     for (uint i = 0; i < 8; i++)
                       node.children[i]->data = node.data | (i << d);
                     return true;
                   });
    }
    /// Destructor
    ~search_tree() override {
      this->clean(this->root_);
      this->root_ = nullptr;
    }
    /// Searches elements inside an given cube in world coordinates
    /// \param bbox search region in world coordinates
    /// \param f callback to process each element found
    void iteratePoints(const BBox &bbox, const std::function<void(uint)> &f) {
      BBox gbbox = zps_.toSet_(bbox);
      traverse([&](Octree<uint>::Node &node) -> bool {
        if (bbox_bbox_intersection(gbbox, node.region())) {
          if (!node.isLeaf())
            return true;
          for (iterator it(zps_, node.data, node.level()); it.next(); ++it)
          if (bbox.contains(it.getWorldPosition()))
            f(it.getId());
        }
        return false;
      });
    }
  private:
    ZPointSet &zps_;
  };
  /// Default constructor
  ZPointSet();
  /// Default destructor
  ~ZPointSet() override;
  /// Constructor
  /// \param maxCoordinates maximum coordinates value for an input point
  explicit ZPointSet(uint maxCoordinates);
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
  /// search points that intersect a bbox
  /// \param b search region, world coordinates
  /// \param f callback to receive the id of each found point
  void search(const BBox &b, const std::function<void(uint)> &f) override;
  //void
  //iteratePoints(const std::function<void(size_t, Point3)> &f) const override;
private:
  /// morton code transform
  /// \return morton code of coordinate **p**
  static uint computeIndex(const Point3 &p);
  search_tree *tree_;   ///< tree for search operations
  Point3 resolution_;   ///< maximum coordinates
  Transform toSet_;     ///< map to underling grid
  std::vector<PointElement> points_; ///< array of point elements
  std::vector<Point3> positions_;    ///< points positions
  std::vector<uint> indices_;        ///< map _points -> _positions
  uint end_;          ///< current number of active point element
  uint lastId_;       ///< last point element id generated
  uint nbits_;        ///< number of bits used by the maximum coordinate component value
  uint maxDepth_;     ///< maximum depth of search tree, bounded by nbits
  uint maxZCode_;     ///< maximum allowed morton code
  bool needUpdate_;   ///< indicates if elements must be sorted
  bool needZUpdate_;  ///< indicates if morton codes need to be recalculated
  bool sizeChanged_;  ///< indicates if points have been added/removed
};

} // ponos namespace

#endif //PONOS_Z_POINT_SET_H
