//
// Created by filipecn on 2018-12-18.
//
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

#ifndef PONOS_BVH_H
#define PONOS_BVH_H

namespace ponos {

enum class BVHSplitMethod { SAH, HLBVH, Middle, EqualCounts };

/// Boundary Volume Hierarchy structure for spatial organization of objects
/// \tparam T object type
template <typename T> class BVH {
public:
  struct ElementInfo {
    ElementInfo(size_t elementNumber, const bbox3 &bounds);
    size_t elementNumber;
    bbox3 bounds;
    point3 centroid;
  };
  struct BuildNode {
    void initLeaf(int first, int n, const bbox3 &b);
    void initInterior(int axis, BuildNode *child0, BuildNode *child1);
    bbox3 bounds;
    BuildNode *children[2];
    int splitAxis{}, firstElementOffset{}, nElements{};
  };
  /// \param o object smart pointer array
  /// \param maxElementsInNode maximum number of elements allowed in a node
  /// \param method split method
  BVH(const std::vector<std::shared_ptr<T>> &o, int maxElementsInNode,
      BVHSplitMethod method);

  BuildNode *recursiveBuild(MemoryArena &arena,
                            std::vector<ElementInfo> &elementInfo, int start,
                            int end, int *totalNodes,
                            std::vector<std::shared_ptr<T>> &orderedElements);

protected:
  const int maxElementsInNode;
  std::vector<std::shared_ptr<T>> elements;
  const BVHSplitMethod splitMethod;
};

#include "bvh.inl"

} // namespace ponos

#endif // PONOS_BVH_H
