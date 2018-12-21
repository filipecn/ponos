template <typename T>
BVH<T>::ElementInfo::ElementInfo(size_t elementNumber,
                                 const ponos::bbox3 &bounds)
    : elementNumber(elementNumber), bounds(bounds),
      centroid(bounds.centroid()) {}

template <typename T>
void BVH<T>::BuildNode::initLeaf(int first, int n, const ponos::bbox3 &b) {
  firstElementOffset = first;
  nElements = n;
  bounds = b;
  children[0] = children[1] = nullptr;
}

template <typename T>
void BVH<T>::BuildNode::initInterior(int axis, BuildNode *child0,
                                     BuildNode *child1) {
  children[0] = child0;
  children[1] = child1;
  bounds = make_union(child0->bounds, child1->bounds);
  splitAxis = axis;
  nElements = 0;
}

template <typename T>
BVH<T>::BVH(const std::vector<std::shared_ptr<T>> &o, int maxElementsInNode,
            BVHSplitMethod method)
    : maxElementsInNode(std::min(255, maxElementsInNode)), elements(o),
      splitMethod(method) {
  if (o.empty())
    return;
  // initialize element info
  std::vector<ElementInfo> elementInfo(o.size());
  for (size_t i = 0; i < elementInfo.size(); ++i)
    elementInfo[i] = {i, elements[i]->worldBound()};
  // build BVH tree for elements using elementInfo
  MemoryArena arena(1024 * 1024);
  int totalNodes = 0;
  std::vector<std::shared_ptr<T>> orderedElements;
  BuildNode *root;
  if (splitMethod == BVHSplitMethod::HLBVH)
    root = HLBVHBuild(arena, elementInfo, &totalNodes, orderedElements);
  else
    root = recursiveBuild(arena, elementInfo, 0, elements.size(), &totalNodes,
                          orderedElements);
  elements.swap(orderedElements);
  // compute representation of depth-first traversal of BVH tree TODO
}

template <typename T>
typename BVH<T>::BuildNode *
BVH<T>::recursiveBuild(MemoryArena &arena,
                       std::vector<ElementInfo> &elementInfo, int start,
                       int end, int *totalNodes,
                       std::vector<std::shared_ptr<T>> &orderedElements) {
  BuildNode *node = arena.alloc<BuildNode>();
  (*totalNodes)++;
  // compute bounds of all elements in BVH node
  bbox3 bounds;
  for (int i = start; i < end; ++i)
    bounds = make_union(bounds, elementInfo[i]);
  int nElements = end - start;
  if (nElements == 1) {
    // create leaf BuildNode
    int firstElementOffset = static_cast<int>(orderedElements.size());
    for (int i = start; i < end; ++i) {
      int elementNumber = elementInfo[i].elementNumber;
      orderedElements.push_back(elements[elementNumber]);
    }
    node->initLeaf(firstElementOffset, nElements, bounds);
    return node;
  } else {
    // compute bound of element centroids, choose split dimension dim
    bbox3 centroidBounds;
    for (int i = start; i < end; ++i)
      centroidBounds = make_union(centroidBounds, elementInfo[i].centroid);
    int dim = centroidBounds.maxExtent();
    // partition elements into two sets and build children
    int mid = (start + end) / 2;
    if (centroidBounds.upper[dim] == centroidBounds.lower[dim]) {
      // create leaf BuildNode
      int firstElementOffset = static_cast<int>(orderedElements.size());
      for (int i = start; i < end; ++i) {
        int elementNumber = elementInfo[i].elementNumber;
        orderedElements.push_back(elements[elementNumber]);
      }
      node->initLeaf(firstElementOffset, nElements, bounds);
      return node;
    } else {
      // partition elements based on split method
      switch (splitMethod) {
      case BVHSplitMethod::SAH: {
        if (nElements <= 4) {
          // partition elements into equally sized subsets
          mid = (start + end) / 2;
          std::nth_element(&elementInfo[start], &elementInfo[mid],
                           &elementInfo[end - 1] + 1,
                           [dim](const ElementInfo &a, const ElementInfo &b) {
                             return a.centroid[dim] < b.centroid[dim];
                           });
        } else {
          // allocate BucketInfo for SAH partition buckets
          constexpr int nBuckets = 12;
          struct BucketInfo {
            int count = 0;
            bbox3 bounds;
          };
          BucketInfo buckets[nBuckets];
          // initialize BucketInfo for SAH partition buckets
          for (int i = start; i < end; ++i) {
            int b =
                nBuckets * centroidBounds.offset(elementInfo[i].centroid)[dim];
            if (b == nBuckets)
              b = nBuckets - 1;
            buckets[b].count++;
            buckets[b].bounds =
                make_union(buckets[b].bounds, elementInfo[i].bounds);
          }
          // compute costs for splitting after each bucket
          real_t cost[nBuckets - 1];
          for (int i = 0; i < nBuckets - 1; ++i) {
            bbox3 b0, b1;
            int count0 = 0, count1 = 0;
            for (int j = 0; j <= i; ++j) {
              b0 = make_union(b0, buckets[j].bounds);
              count0 += buckets[j].count;
            }
            for (int j = i + 1; j < nBuckets; ++j) {
              b1 = make_union(b1, buckets[j].bounds);
              count1 += buckets[j].count;
            }
            cost[i] = .125f +
                      (count0 * b0.surfaceArea() + count1 * b1.surfaceArea()) /
                          bounds.surfaceArea();
          }
          // find bucket to split at that minimizes SAH metric
          real_t minCost = cost[0];
          int minCostSplitBucket = 0;
          for (int i = 1; i < nBuckets - 1; i++)
            if (cost[i] < minCost) {
              minCost = cost[i];
              minCostSplitBucket = i;
            }
          // either create leaf or split elements at selected SAH bucket
          real_t leafCost = nElements;
          if (nElements > maxElementsInNode || minCost < leafCost) {
            ElementInfo *pmid = std::partition(
                &elementInfo[start], &elementInfo[end] + 1,
                [=](const ElementInfo &info) {
                  int b = nBuckets * centroidBounds.offset(info.centroid)[dim];
                  if (b == nBuckets)
                    b = nBuckets - 1;
                  return b <= minCostSplitBucket;
                });
            mid = pmid - &elementInfo[0];
          } else {
            int firstElementOffset = static_cast<int>(orderedElements.size());
            for (int i = start; i < end; ++i) {
              int elementNumber = elementInfo[i].elementNumber;
              orderedElements.push_back(elements[elementNumber]);
            }
            node->initLeaf(firstElementOffset, nElements, bounds);
            return node;
          }
        }
      } break;
      case BVHSplitMethod::Middle: {
        real_t pmid =
            (centroidBounds.lower[dim] + centroidBounds.upper[dim]) / 2;
        ElementInfo *midPtr =
            std::partition(&elementInfo[start], &elementInfo[end - 1] + 1,
                           [dim, pmid](const ElementInfo &pi) {
                             return pi.centroid[dim] < pmid;
                           });
        mid = midPtr - &elementInfo[0];
        if (mid != start && mid != end)
          break;
      }
      case BVHSplitMethod::EqualCounts: {
        mid = (start + end) / 2;
        std::nth_element(&elementInfo[start], &elementInfo[mid],
                         &elementInfo[end - 1] + 1,
                         [dim](const ElementInfo &a, const ElementInfo &b) {
                           return a.centroid[dim] < b.centroid[dim];
                         });
      } break;
      }
      node->initInterior(dim,
                         recursiveBuild(arena, elementInfo, start, mid,
                                        totalNodes, orderedElements),
                         recursiveBuild(arena, elementInfo, mid, end,
                                        totalNodes, orderedElements));
    }
  }
  return node;
}
