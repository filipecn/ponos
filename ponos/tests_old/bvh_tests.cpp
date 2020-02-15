//
// Created by filipecn on 2018-12-20.
//

#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;

TEST(BVH, Build) {
  struct O {
    bbox3 worldBound() { return b; }
    bbox3 b;
  };
  std::vector<std::shared_ptr<O>> objects;
  BVH<O> bvh(objects, 1, BVHSplitMethod::EqualCounts);
}