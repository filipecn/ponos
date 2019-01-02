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

#include <ponos/geometry/bbox.h>
#include <ponos/parallel/parallel.h>

namespace ponos {

enum class BVHSplitMethod { SAH, HLBVH, Middle, EqualCounts };

/// Boundary Volume Hierarchy structure for spatial organization of objects
/// \tparam T object type
template <typename T> class BVH {
public:
  struct ElementInfo {
    ElementInfo() = default;
    ElementInfo(size_t elementNumber, const bbox3f &bounds);
    size_t elementNumber;
    bbox3f bounds;
    point3f centroid;
  };
  struct BuildNode {
    void initLeaf(int first, int n, const bbox3f &b);
    void initInterior(int axis, BuildNode *child0, BuildNode *child1);
    bbox3f bounds;
    BuildNode *children[2];
    int splitAxis{}, firstElementOffset{}, nElements{};
  };
  struct MortonElement {
    int elementIndex;
    uint32_t mortonCode;
  };
  struct LBVHTreelet {
    int startIndex, nElements;
    BuildNode *buildNodes;
  };
  struct LinearNode {
    bbox3f bounds;
    union {
      int elementOffset;     // leaf
      int secondChildOffset; // interior
    };
    uint16_t nElements; // 0 -> interior node
    uint8_t axis;       // interior node: xyz
    uint8_t pad[1];     // ensure 32 byte total size
  };
  /// \param o object smart pointer array
  /// \param maxElementsInNode maximum number of elements allowed in a node
  /// \param method split method
  BVH(const std::vector<std::shared_ptr<T>> &o, int maxElementsInNode,
      BVHSplitMethod method);

protected:
  BuildNode *recursiveBuild(MemoryArena &arena,
                            std::vector<ElementInfo> &elementInfo, int start,
                            int end, int *totalNodes,
                            std::vector<std::shared_ptr<T>> &orderedElements);
  BuildNode *HLBVHBuild(MemoryArena &arena,
                        std::vector<ElementInfo> &elementInfo, int *totalNodes,
                        std::vector<std::shared_ptr<T>> &orderedElements);
  BuildNode *
  emitLBVH(BuildNode *&buildNodes, const std::vector<ElementInfo> &elementInfo,
           MortonElement *mortonElements, int nElements, int *totalNodes,
           std::vector<std::shared_ptr<T>> &orderedElements,
           std::atomic<int> *orderedELementsOffset, int bitIndex) const {
    if (bitIndex == -1 || nElements < maxElementsInNode) {
      // create and return leaf node of LBVH treelet
      (*totalNodes)++;
      BuildNode *node = buildNodes++;
      bbox3f bounds;
      int firstElementOffset = orderedELementsOffset->fetch_add(nElements);
      for (int i = 0; i < nElements; ++i) {
        int elementIndex = mortonElements[i].elementIndex;
        orderedElements[firstElementOffset + i] = elements[elementIndex];
        bounds = make_union(bounds, elementInfo[elementIndex].bounds);
      }
      node->initLeaf(firstElementOffset, nElements, bounds);
      return node;
    } else {
      int mask = 1 << bitIndex;
      // advance to next subtree level if there's no LBVH split for hits bit
      if ((mortonElements[0].mortonCode & mask) ==
          (mortonElements[nElements - 1].mortonCode & mask))
        return emitLBVH(buildNodes, elementInfo, mortonElements, nElements,
                        totalNodes, orderedElements, orderedELementsOffset,
                        bitIndex - 1);
      // find LBVH split point for this dimension
      int searchStart = 0, searchEnd = nElements - 1;
      while (searchStart + 1 != searchEnd) {
        int mid = (searchStart + searchEnd) / 2;
        if ((mortonElements[searchStart].mortonCode & mask) ==
            (mortonElements[mid].mortonCode & mask))
          searchStart = mid;
        else
          searchEnd = mid;
      }
      int splitOffset = searchEnd;
      // create and return interior LBVH node
      (*totalNodes)++;
      BuildNode *node = buildNodes++;
      BuildNode *lbvh[2] = {emitLBVH(buildNodes, elementInfo, mortonElements,
                                     splitOffset, totalNodes, orderedElements,
                                     orderedELementsOffset, bitIndex - 1),
                            emitLBVH(buildNodes, elementInfo, mortonElements,
                                     nElements - splitOffset, totalNodes,
                                     orderedElements, orderedELementsOffset,
                                     bitIndex - 1)};
      int axis = bitIndex % 3;
      node->initInterior(axis, lbvh[0], lbvh[1]);
      return node;
    }
  }
  BuildNode *buildUpperSAH(MemoryArena &arena,
                           std::vector<BuildNode *> &treeletRoots, int start,
                           int end, int *totalNodes) const {
    int nNodes = end - start;
    if (nNodes == 1)
      return treeletRoots[start];
    (*totalNodes)++;
    BuildNode *node = arena.alloc<BuildNode>();
    // Compute bounds of all nodes under this HLBVH node
    bbox3f bounds;
    for (int i = start; i < end; ++i)
      bounds = make_union(bounds, treeletRoots[i]->bounds);

    // Compute bound of HLBVH node centroids, choose split dimension _dim_
    bbox3f centroidBounds;
    for (int i = start; i < end; ++i) {
      point3f centroid =
          (treeletRoots[i]->bounds.lower + treeletRoots[i]->bounds.upper) *
          0.5f;
      centroidBounds = make_union(centroidBounds, centroid);
    }
    int dim = centroidBounds.maxExtent();
    // Allocate _BucketInfo_ for SAH partition buckets
    constexpr int nBuckets = 12;
    struct BucketInfo {
      int count = 0;
      bbox3f bounds;
    };
    BucketInfo buckets[nBuckets];

    // Initialize _BucketInfo_ for HLBVH SAH partition buckets
    for (int i = start; i < end; ++i) {
      real_t centroid = (treeletRoots[i]->bounds.lower[dim] +
                         treeletRoots[i]->bounds.upper[dim]) *
                        0.5f;
      int b =
          nBuckets * ((centroid - centroidBounds.lower[dim]) /
                      (centroidBounds.upper[dim] - centroidBounds.lower[dim]));
      if (b == nBuckets)
        b = nBuckets - 1;
      buckets[b].count++;
      buckets[b].bounds =
          make_union(buckets[b].bounds, treeletRoots[i]->bounds);
    }

    // Compute costs for splitting after each bucket
    real_t cost[nBuckets - 1];
    for (int i = 0; i < nBuckets - 1; ++i) {
      bbox3f b0, b1;
      int count0 = 0, count1 = 0;
      for (int j = 0; j <= i; ++j) {
        b0 = make_union(b0, buckets[j].bounds);
        count0 += buckets[j].count;
      }
      for (int j = i + 1; j < nBuckets; ++j) {
        b1 = make_union(b1, buckets[j].bounds);
        count1 += buckets[j].count;
      }
      cost[i] =
          .125f + (count0 * b0.surfaceArea() + count1 * b1.surfaceArea()) /
                      bounds.surfaceArea();
    }

    // Find bucket to split at that minimizes SAH metric
    real_t minCost = cost[0];
    int minCostSplitBucket = 0;
    for (int i = 1; i < nBuckets - 1; ++i) {
      if (cost[i] < minCost) {
        minCost = cost[i];
        minCostSplitBucket = i;
      }
    }

    // Split nodes and create interior HLBVH SAH node
    BuildNode **pmid = std::partition(
        &treeletRoots[start], &treeletRoots[end - 1] + 1,
        [=](const BuildNode *node) {
          real_t centroid =
              (node->bounds.lower[dim] + node->bounds.upper[dim]) * 0.5f;
          int b = nBuckets *
                  ((centroid -
                    static_cast<const point3>(centroidBounds.lower)[dim]) /
                   (static_cast<const point3>(centroidBounds.upper)[dim] -
                    static_cast<const point3>(centroidBounds.lower)[dim]));
          if (b == nBuckets)
            b = nBuckets - 1;
          // CHECK_GE(b, 0);
          // CHECK_LT(b, nBuckets);
          return b <= minCostSplitBucket;
        });
    int mid = pmid - &treeletRoots[0];
    // CHECK_GT(mid, start);
    // CHECK_LT(mid, end);
    node->initInterior(
        dim, this->buildUpperSAH(arena, treeletRoots, start, mid, totalNodes),
        this->buildUpperSAH(arena, treeletRoots, mid, end, totalNodes));
    return node;
  }
  void radixSort(std::vector<MortonElement> *v);
  int flattenBVHTree(BuildNode *node, int *offset);

  const int maxElementsInNode;
  std::vector<std::shared_ptr<T>> elements;
  const BVHSplitMethod splitMethod;
  LinearNode *nodes = nullptr;
};

#include "bvh.inl"

} // namespace ponos

#endif // PONOS_BVH_H
