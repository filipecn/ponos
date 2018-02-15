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
  resolution_ = Point3(maxCoordinates, maxCoordinates, maxCoordinates);
  // TODO: solve this
  //auto h = 1.f / maxCoordinates;
  //toSet_ = scale(h, h, h);
  maxZCode_ = computeIndex(resolution_);
  int n = maxCoordinates - 1;
  for (nbits_ = 0; n; n >>= 1)
    nbits_++;
  maxDepth_ = nbits_;
}

void ZPointSet::update() {
  if (!points_.size())
    return;
  if (needZUpdate_) {
    for (uint i = 0; i < end_; i++) {
      Point3 gp = toSet_(positions_[points_[i].id]);
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
  if (tree_)
    delete tree_;
  tree_ = new search_tree(*this);
  needUpdate_ = false;
  sizeChanged_ = false;
}

uint ZPointSet::size() {
  if (sizeChanged_)
    update();
  return end_;
}

size_t ZPointSet::add(Point3 p) {
  if (end_ == points_.size()) {
    points_.emplace_back();
    points_[end_].id = lastId_++;
  }
  if (points_[end_].id == positions_.size())
    positions_.emplace_back();
  points_[end_].active = true;
  positions_[points_[end_].id] = p;
  Point3 sp = toSet_(p);
  FATAL_ASSERT(sp >= Point3() && sp <= resolution_);
  points_[end_].zcode = computeIndex(sp);
  FATAL_ASSERT(points_[end_].zcode <= maxZCode_);
  end_++;
  needUpdate_ = true;
  sizeChanged_ = true;
  return points_[end_ - 1].id;
}

void ZPointSet::setPosition(unsigned int i, Point3 p) {
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

Point3 ZPointSet::operator[](uint i) const {
  FATAL_ASSERT(i < positions_.size());
  return positions_[i];
}

uint ZPointSet::computeIndex(const Point3 &p) {
  return mortonCode(static_cast<uint32_t>(p.x), static_cast<uint32_t>(p.y), static_cast<uint32_t>(p.z));
}

void ZPointSet::search(const BBox &b, const std::function<void(uint)> &f) {
  if (!tree_ || needUpdate_ || needZUpdate_)
    update();
  if (!tree_)
    return;
  tree_->iteratePoints(b, [&](uint id) { f(id); });
}

} // ponos namespace
