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

#include "z_point_set.h"

namespace ponos {

ZPointSet::ZPointSet() {
  tree_ = nullptr;
  end_ = 0;
  lastId_ = 0;
  nbits_ = 0;
  maxDepth_ = 0;
  maxZCode_ = 0;
  needUpdate_ = true;
  needZUpdate_ = true;
  sizeChanged_ = true;
};

ZPointSet::~ZPointSet() = default;

ZPointSet::ZPointSet(uint maxCoordinates) : ZPointSet() {
  FATAL_ASSERT(isPowerOf2(maxCoordinates));
  resolution_ = point3(maxCoordinates, maxCoordinates, maxCoordinates);
  // TODO: solve this
  // auto h = 1.f / maxCoordinates;
  // toSet_ = scale(h, h, h);
  maxZCode_ = computeIndex(resolution_);
  int n = maxCoordinates - 1;
  for (nbits_ = 0; n; n >>= 1)
    nbits_++;
  maxDepth_ = nbits_;
}

void ZPointSet::update() {
  if (points_.empty())
    return;
  if (needZUpdate_) {
    for (uint i = 0; i < end_; i++) {
      point3 gp = toSet_(positions_[points_[i].id]);
      points_[i].zcode = computeIndex(gp);
      FATAL_ASSERT(points_[i].zcode <= maxZCode_);
    }
    needZUpdate_ = false;
    needUpdate_ = true;
  }
  if (!needUpdate_)
    return;
  std::sort(&points_[0], &points_[0] + end_,
            [](const PointElement &a, const PointElement &b) {
              if (!a.active)
                return false;
              if (!b.active)
                return true;
              return a.zcode < b.zcode;
            });
  if (indices_.size() < end_)
    indices_.resize(end_);
  for (size_t i = 0; i < end_; i++)
    indices_[points_[i].id] = i;
  // compute new end in case some points have been deleted
  if (end_ > 0) {
    while (end_ > 0 && !points_[end_ - 1].active)
      end_--;
  }
  needUpdate_ = false;
  sizeChanged_ = false;
}

void ZPointSet::buildAccelerationStructure() {
  if (tree_)
    delete tree_;
  update();
  tree_ = new search_tree(*this);
}

uint ZPointSet::size() {
  if (sizeChanged_)
    update();
  return end_;
}

uint ZPointSet::add(point3 p) {
  if (end_ == points_.size()) {
    points_.emplace_back();
    points_[end_].id = lastId_++;
  }
  if (points_[end_].id == positions_.size())
    positions_.emplace_back();
  points_[end_].active = true;
  positions_[points_[end_].id] = p;
  point3 sp = toSet_(p);
  FATAL_ASSERT(sp >= point3() && sp <= resolution_);
  points_[end_].zcode = computeIndex(sp);
  FATAL_ASSERT(points_[end_].zcode <= maxZCode_);
  end_++;
  needUpdate_ = true;
  sizeChanged_ = true;
  return points_[end_ - 1].id;
}

void ZPointSet::setPosition(uint i, point3 p) {
  positions_[i] = p;
  needZUpdate_ = true;
  needUpdate_ = true;
}

void ZPointSet::remove(uint i) {
  FATAL_ASSERT(i < indices_.size());
  FATAL_ASSERT(indices_[i] < points_.size());
  points_[indices_[i]].active = false;
  needUpdate_ = true;
  sizeChanged_ = true;
}

point3 ZPointSet::operator[](uint i) const {
  FATAL_ASSERT(i < positions_.size());
  return positions_[i];
}

uint ZPointSet::computeIndex(const point3 &p) {
  return encodeMortonCode(static_cast<uint32_t>(p.x),
                          static_cast<uint32_t>(p.y),
                          static_cast<uint32_t>(p.z));
}

void ZPointSet::search(const bbox3 &b, const std::function<void(uint)> &f) {
  if (!tree_ || needUpdate_ || needZUpdate_)
    update();
  if (tree_) {
    tree_->iteratePoints(b, [&](uint id) { f(id); });
    return;
  }
  // perform an implicit search
  std::function<void(uint, uint, const bbox3 &, const bbox3 &,
                     const std::function<void(uint)> &)>
      implictTraverse = [&](uint level, uint zcode, const bbox3 &region,
                            const bbox3 &bbox,
                            const std::function<void(uint)> &callBack) {
        if (level >= maxDepth_)
          return;
        // check if node is fully contained by bbox or node is leaf, refine
        // otherwise
        if (bbox.contains(region) || level == maxDepth_ - 1) {
          if (bbox_bbox_intersection(bbox, region))
            for (iterator it(*this, zcode, level); it.next(); ++it)
              if (bbox.contains(it.getWorldPosition()))
                f(it.getId());
          return;
        }
        uint d = (nbits_ - level - 1) * 3;
        auto regions = region.splitBy8();
        for (uint i = 0; i < 8; i++)
          implictTraverse(level + 1, zcode | (i << d), regions[i], bbox,
                          callBack);
      };
  implictTraverse(0, 0, bbox3(point3(), resolution_), b, f);
}

void ZPointSet::iteratePoints(
    const std::function<void(uint, point3)> &f) const {
  for (uint i = 0; i < end_; i++)
    f(points_[i].id, positions_[points_[i].id]);
}

int ZPointSet::intersect(const Ray3 &r, float e) {
  float minDistance2 = 1 << 20;
  int closestPoint = -1;
  // perform an implicit search
  std::function<void(uint, uint, const bbox3 &)> implictTraverse =
      [&](uint level, uint zcode, const bbox3 &region) {
        if (level >= maxDepth_)
          return;
        float hit1 = 0, hit2 = 0;
        if (!bbox_ray_intersection(region, r, hit1, hit2))
          return;
        // TODO check if hit is < minDistance2 to prune far regions
        if (level == maxDepth_ - 1) {
          for (iterator it(*this, zcode, level); it.next(); ++it)
            if (distance_point_line(it.getWorldPosition(), Line(r.o, r.d)) <
                e) {
              float dist = distance2(r.o, it.getWorldPosition());
              if (dist < minDistance2) {
                closestPoint = it.getId();
                minDistance2 = dist;
              }
            }
        } else {
          uint d = (nbits_ - level - 1) * 3;
          auto regions = region.splitBy8();
          for (uint i = 0; i < 8; i++)
            implictTraverse(level + 1, zcode | (i << d), regions[i]);
        }
      };
  implictTraverse(0, 0, bbox3(point3(), resolution_));
  return closestPoint;
}

void ZPointSet::cast(const Ray3 &r, const std::function<void(uint)> &f) {
  UNUSED_VARIABLE(r);
  UNUSED_VARIABLE(f);
}

} // namespace ponos
