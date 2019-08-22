template <typename T>
BVH<T>::ElementInfo::ElementInfo(size_t elementNumber,
                                 const ponos::bbox3f &bounds)
    : elementNumber(elementNumber), bounds(bounds),
      centroid(bounds.centroid()) {}

template <typename T>
void BVH<T>::BuildNode::initLeaf(int first, int n, const bbox3f &b) {
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
  // compute representation of depth-first traversal of BVH tree
  nodes = allocAligned<LinearNode>(totalNodes);
  int offset = 0;
  flattenBVHTree(root, &offset);
}

template <typename T>
int BVH<T>::flattenBVHTree(BVH<T>::BuildNode *node, int *offset) {
  LinearNode *linearNode = &nodes[*offset];
  linearNode->bounds = node->bounds;
  int myOffset = (*offset)++;
  if (node->nElements > 0) {
    linearNode->elementOffset = node->firstElementOffset;
    linearNode->nElements = node->nElements;
  } else {
    // create interior flattene BVH node
    linearNode->axis = node->splitAxis;
    linearNode->nElements = 0;
    flattenBVHTree(node->children[0], offset);
    linearNode->secondChildOffset = flattenBVHTree(node->children[1], offset);
  }
  return myOffset;
}

template <typename T>
typename BVH<T>::BuildNode *
BVH<T>::recursiveBuild(MemoryArena &arena,
                       std::vector<BVH<T>::ElementInfo> &elementInfo, int start,
                       int end, int *totalNodes,
                       std::vector<std::shared_ptr<T>> &orderedElements) {
  BuildNode *node = arena.alloc<BuildNode>();
  (*totalNodes)++;
  // compute bounds of all elements in BVH node
  bbox3f bounds;
  for (int i = start; i < end; ++i)
    bounds = make_union(bounds, elementInfo[i].bounds);
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
    bbox3f centroidBounds;
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
            bbox3f bounds;
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

template <typename T>
void BVH<T>::radixSort(std::vector<BVH<T>::MortonElement> *v) {
  std::vector<MortonElement> tempVector(v->size());
  constexpr int bitsPerPass = 6;
  constexpr int nBits = 30;
  constexpr int nPasses = nBits / bitsPerPass;
  for (int pass = 0; pass < nPasses; ++pass) {
    // perform one pass of radix sort, sorting bitsPerPass bits
    int lowBit = pass * bitsPerPass;
    // set in and out vector pointers for radix sort pass
    std::vector<MortonElement> &in = (pass & 1) ? tempVector : *v;
    std::vector<MortonElement> &out = (pass & 1) ? *v : tempVector;
    // count number of zero bits in array for current radix sort bit
    constexpr int nBuckets = 1 << bitsPerPass;
    int bucketCount[nBuckets] = {0};
    constexpr int bitMask = (1 << bitsPerPass) - 1;
    for (const MortonElement &me : in) {
      int bucket = (me.mortonCode >> lowBit) & bitMask;
      ++bucketCount[bucket];
    }
    // compute starting index in output array for each bucket
    int outIndex[nBuckets];
    outIndex[0] = 0;
    for (int i = 1; i < nBuckets; ++i)
      outIndex[i] = outIndex[i - 1] + bucketCount[i - 1];
    // store sorted values in output array
    for (const MortonElement &me : in) {
      int bucket = (me.mortonCode >> lowBit) & bitMask;
      out[outIndex[bucket]++] = me;
    }
  }
  // copy final result from tempVector, if needed
  if (nPasses & 1)
    std::swap(*v, tempVector);
}

template <typename T>
typename BVH<T>::BuildNode *
BVH<T>::HLBVHBuild(MemoryArena &arena, std::vector<ElementInfo> &elementInfo,
                   int *totalNodes,
                   std::vector<std::shared_ptr<T>> &orderedElements) {
  // compute bbox of all element centroids
  bbox3f bounds;
  for (const ElementInfo &ei : elementInfo)
    bounds = make_union(bounds, ei.centroid);
  // compute morton indices of elements
  std::vector<MortonElement> mortonElements(elementInfo.size());
  parallel_for(
      [&](int i) {
        // initialize mortonElements[i] for i th element
        constexpr int mortonBits = 10;
        constexpr int mortonScale = 1 << mortonBits;
        mortonElements[i].elementIndex = elementInfo[i].elementNumber;
        vec3 centroidOffset = bounds.offset(elementInfo[i].centroid) *
                              static_cast<real_t>(mortonScale);
        mortonElements[i].mortonCode = encodeMortonCode(
            centroidOffset.x, centroidOffset.y, centroidOffset.z);
      },
      elementInfo.size(), 512);
  // radix sort element morton indices
  radixSort(&mortonElements);
  // create LBVH treelets at bottom of BVH
  //  find interval of elements for each treelet
  std::vector<LBVHTreelet> treeletsToBuild;
  for (int start = 0, end = 1; end <= (int)mortonElements.size(); ++end) {
    uint32_t mask = 0b00111111111111000000000000000000;
    if ((end == (int)mortonElements.size()) ||
        ((mortonElements[start].mortonCode & mask) !=
         (mortonElements[end].mortonCode & mask))) {
      // add entry to treeletsToBuild dor this treelet
      int nElements = end - start;
      int maxBVHNodes = 2 * nElements;
      BuildNode *nodes = arena.alloc<BuildNode>(maxBVHNodes /*, false TODO*/);
      treeletsToBuild.push_back({start, nElements, nodes});
      start = end;
    }
  }
  //  create LBVHs for treelets in parallel
  std::atomic<int> atomicTotal(0), orderedElementsOffset(0);
  orderedElements.resize(elements.size());
  parallel_for(
      [&](int i) {
        // generate ith LBVH treelet
        int nodesCreated = 0;
        const int firstBitIndex = 29 - 12;
        LBVHTreelet &tr = treeletsToBuild[i];
        tr.buildNodes =
            emitLBVH(tr.buildNodes, elementInfo, &mortonElements[tr.startIndex],
                     tr.nElements, &nodesCreated, orderedElements,
                     &orderedElementsOffset, firstBitIndex);
        atomicTotal += nodesCreated;
      },
      treeletsToBuild.size());
  *totalNodes = atomicTotal;
  // create and return SAH BVH from LBVH treelets
  std::vector<BuildNode *> finishedTreelets;
  for (LBVHTreelet &treelet : treeletsToBuild)
    finishedTreelets.push_back(treelet.buildNodes);
  return buildUpperSAH(arena, finishedTreelets, 0, finishedTreelets.size(),
                       totalNodes);
}
