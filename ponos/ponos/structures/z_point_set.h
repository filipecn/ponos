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
  struct NodeElement {
    uint zcode; ///< morton code of the first position this node can contain
    uint pointIndex; ///< first point present on this node
  };
  /// Helper class for iterating elements
  class iterator {
  public:
    /// \param s z point set reference
    /// \param f first element id (morton code)
    /// \param depth tree level
    explicit iterator(ZPointSet &s, uint f = 0, uint depth = 0) :
        lastIndex_(-1), first_(f), depth_(depth), cur_(0), upperBound_(0), zps_(s) {
      if (zps_.end_ == 0)
        return;
      firstIndex_ = find(s, 0, s.end_, first_, depth, &upperBound_);
      cur_ = firstIndex_;
      FATAL_ASSERT(lastIndex_ <= static_cast<int>(zps_.end_));
    }
    /// Given a range of indices from points_ array, finds the first element obeying
    /// f and depth restrictions. It performs a lower_bound operation, so the element
    /// may not be valid.
    /// \param z z point set reference
    /// \param l first index of points_ array
    /// \param s size of range on points_array
    /// \param f first element id (morton code)
    /// \param depth tree level
    /// \param upperBound **[out | optional]** upper bound morton code for this search
    static uint find(ZPointSet &z, uint l, uint s, uint f, uint depth, uint *upperBound = nullptr) {
      std::function<int(const PointElement &p, const uint &v)> comp;
      comp = [](const PointElement &p, const uint &v) {
        if (p.zcode < v)
          return -1;
        if (p.zcode > v)
          return 1;
        return 0;
      };
      FATAL_ASSERT(z.end_ >= l + s);
      if (upperBound)
        *upperBound = f + (1 << ((z.nbits_ - depth) * 3));
      FATAL_ASSERT(depth <= z.maxDepth_);
      return lower_bound<PointElement, uint>(&z.points_[l], s, f, comp) + l + 1;
    }
    /// \return true if there is more elements to iterate
    bool next() const {
      return cur_ < static_cast<int>(zps_.end_) &&
          zps_.points_[cur_].zcode < upperBound_;
    }
    /// \return position of the current element in world coordinates
    Point3 getWorldPosition() {
      FATAL_ASSERT(cur_ < static_cast<int>(zps_.points_.size()));
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
      FATAL_ASSERT(cur_ < static_cast<int>(zps_.points_.size()));
      return zps_.points_[cur_].id;
    }
    /// \return pointer to current point element struct
    PointElement *pointElement() {
      FATAL_ASSERT(cur_ < static_cast<int>(zps_.points_.size()));
      return &zps_.points_[cur_];
    }
    /// advance on iteration
    void operator++() { cur_++; }
    /// \return number of elements to iterate (first -> last)
    uint count() {
      if (lastIndex_ < 1) {
        lastIndex_ = find(zps_, static_cast<uint>(firstIndex_), zps_.end_ - firstIndex_, upperBound_, depth_);
        FATAL_ASSERT(lastIndex_ >= firstIndex_);
      }
      return static_cast<uint>(lastIndex_ - firstIndex_);
    }
  private:
    int firstIndex_;  ///< first element index
    int lastIndex_;   ///< last element index (lazily computed on count())
    uint first_;      ///< morton code of first element
    uint depth_;      ///< depth of node where points must reside
    int cur_;         ///< current element index
    uint upperBound_; ///< upper bound for search (morton code)
    ZPointSet &zps_;
  };
  /// Helper class for search queries
  class search_tree : public Octree<NodeElement> {
  public:
    /// \param z z point set reference
    /// \param f **[optional]** refinement criteria
    explicit search_tree(ZPointSet &z, std::function<bool(uint, uint)> f =
    [](uint id, uint depth) -> bool {
      UNUSED_VARIABLE(id);
      UNUSED_VARIABLE(depth);
      return true;
    })
        : Octree<NodeElement>(), zps_(z) {
      this->root_ = new Octree<NodeElement>::Node(BBox(Point3(), zps_.resolution_));
      this->count_++;
      this->root_->data.zcode = 0;
      this->refine(this->root_,
                   [&](Octree<NodeElement>::Node &node) -> bool {
                     if (node.level() >= zps_.maxDepth_)
                       return false;
                     return f(node.data.zcode, node.level());
                   },
                   [&](Octree<NodeElement>::Node &node) {
                     uint d = (zps_.nbits_ - node.level() - 1) * 3;
                     for (uint i = 0; i < 8; i++)
                       node.children[i]->data.zcode = node.data.zcode | (i << d);
                     return true;
                   });
      if (zps_.end_)
        computePointsIndices();
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
      traverse([&](Octree<NodeElement>::Node &node) -> bool {
        if (gbbox.contains(node.region()) || bbox_bbox_intersection(gbbox, node.region())) {
          uint upperBound = node.data.zcode + (1 << ((zps_.nbits_ - node.level()) * 3));
          for (uint i = node.data.pointIndex; i < zps_.end_ &&
              zps_.points_[i].zcode >= node.data.zcode &&
              zps_.points_[i].zcode < upperBound; ++i)
            if (bbox.contains(zps_.positions_[zps_.points_[i].id]))
              f(zps_.points_[i].id);
          return false;
        }
        return true;
      });
    }
    /// mark point indices on tree nodes for fast point retrieval
    void computePointsIndices() {
      std::function<void(Octree<NodeElement>::Node *, uint, uint)>
          f = [&](Octree<NodeElement>::Node *node, uint l, uint s) {
        UNUSED_VARIABLE(l);
        UNUSED_VARIABLE(s);
        if (!node)
          return;
        node->data.pointIndex = iterator::find(zps_, l, s, node->data.zcode, node->level());
        // TODO: it can ve optimized by considering the point index of child n-1 for child n
        for (auto &i : node->children)
          f(i, node->data.pointIndex, zps_.end_ - node->data.pointIndex);
      };
      f(this->root_, 0, zps_.end_);
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
  /// setup an internal search tree to accelerate point search
  void buildAccelerationStructure();
  // INTERFACE

  uint size() override;
  uint add(Point3 p) override;
  void setPosition(uint i, Point3 p) override;
  void remove(uint i) override;
  Point3 operator[](uint i) const override;
  void search(const BBox &b, const std::function<void(uint)> &f) override;
  void iteratePoints(const std::function<void(uint, Point3)> &f) const override;
  int intersect(const Ray3& r, float e) override;
  void cast(const Ray3& r, const std::function<void(uint)>& f) override;
private:
  /// morton code transform
  /// \return morton code of coordinate **p**
  static uint computeIndex(const Point3 &p);
  search_tree *tree_;   ///< tree for search operations
  Point3 resolution_;   ///< maximum coordinates
  Transform toSet_;     ///< map to underling grid
  std::vector<PointElement> points_; ///< array of point elements
  std::vector<Point3> positions_;    ///< points positions
  std::vector<uint> indices_;        ///< map point id -> points_
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
