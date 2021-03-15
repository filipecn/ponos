/// Copyright (c) 2018, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
///\file bvh.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2018-12-18
///
///\brief Bounding Volume Hierarchies (BVHs) are acceleration structures based on
///       primitive subdivision, where primitives are partitioned i32o a hierarchy
///       of disjoi32 sets. Primitives are stored in the leaves, and each node
///       stores a bounding box of the primitives in the nodes beneath it.
///
///       This implementation is based on the code presented in PBR book.

#ifndef PONOS_BVH_H
#define PONOS_BVH_H

#include <ponos/geometry/bbox.h>
#include <ponos/parallel/parallel.h>

namespace ponos {

/// Algorithms used when partitioning primitives
enum class BVHSplitMethod {
  /// Surface Area Heuristic:
  /// Minimizes total cost (traversal/intersection) by estimating the computational
  /// expense of performing ray intersection tests, including traversal time and
  /// primitive intersection tests. Usually creates the most efficient trees for rendering.
  SAH,
  /// Hierarchical Linear Bounding Volume Hierarchy
  /// Morton-curve based clustering is used to first build trees for the lower levels of the
  /// hierarchy and the top levels of the tree are then created using the SAH. In this
  /// approach the BVH is built using only the centroids of primitive bounding boxes, so it
  /// doesn't account for the actual spatial extent of each primitive.
  HLBVH,
  /// Middle:
  /// Partition primitives based on the midpoint position of the bounding centroids extent
  Middle,
  /// Equal Counts:
  /// Partition primitives into equally sized subsets
  EqualCounts
};

/// Boundary Volume Hierarchy acceleration structure
/// \tparam T primitive (wrapper) type
///         T must provide a method: bbox3 bounds();
template<typename T> class BVH {
public:
  /// Stores bounding and index information about each primitive
  /// to be used in tree construction
  struct ElementInfo {
    ElementInfo() = default;
    /// \param element_number index in the input primitive array
    /// \param bounds primitive bounds
    ElementInfo(u64 element_number, const bbox3f &bounds)
        : element_number(element_number), bounds(bounds),
          centroid(bounds.centroid()) {}
    u64 element_number{0}; //!< index in the input primitive array
    bbox3 bounds; //!< primitive bounds
    point3 centroid; //!< bounds centroid
  };
  /// Represents a node in the BVH
  struct BuildNode {
    /// Interior Node
    /// \param first index of the first primitive in the leaf
    /// \param n number of primitives in the leaf
    /// \param b bounds of all primitives in the leaf
    void initLeaf(i32 first, i32 n, const bbox3f &b) {
      first_element_offset = first;
      n_elements = n;
      bounds = b;
      children[0] = children[1] = nullptr;
    }
    /// Leaf Node
    /// \param axis coordinate axis which primitives are partitioned
    /// \param child0 left child pointer
    /// \param child1 right child pointer
    void initInterior(i32 axis, BuildNode *child0, BuildNode *child1) {
      children[0] = child0;
      children[1] = child1;
      bounds = make_union(child0->bounds, child1->bounds);
      split_axis = axis;
      n_elements = 0;
    }
    bbox3 bounds; //!< bounds of all of the children beneath this node
    BuildNode *children[2]; //!< child pointers (used in interior nodes)
    i32 split_axis{}; //!< coordinate axis which primitives were partitioned
    i32 first_element_offset{}; //!< index of first primitive in this leaf
    i32 n_elements{}; //!< number of primitives in this leaf
  };
  /// Morton code information for each primitive
  struct MortonElement {
    i32 element_index; //!< primitive index in element info array
    u32 morton_code; //!< Morton code of the primitive
  };
  /// LBVH Treelet
  struct LBVHTreelet {
    i32 start_index; //!< first primitive index in morton encoded array
    i32 n_elements; //!< number of primitives inside the treelet
    BuildNode *build_nodes; //!<
  };
  ///
  struct LinearNode {
    bbox3f bounds;
    union {
      i32 element_offset;     // leaf
      i32 second_child_offset; // interior
    };
    u32 n_elements{}; // 0 -> interior node
    u32 axis{};       // interior node: xyz
    u32 pad[1]{};     // ensure 32 byte total size
  };
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  /// \param primitive_list primitives
  /// \param max_elements_in_node maximum number of elements_ allowed in a node
  /// \param method algorithm used when partitioning primitives
  explicit BVH(const std::vector<T> &primitive_list, i32 max_elements_in_node = 1,
               BVHSplitMethod method = BVHSplitMethod::SAH)
      : max_elements_in_node_(std::min(255, max_elements_in_node)),
        elements_(primitive_list),
        split_method_(method) {
    if (primitive_list.empty())
      return;
    // Construction goes as follows:
    //  1) Compute bounding information about each primitive and store in an array
    //  2) Build the tree using the given choice of split algorithm
    //  3) Convert the resulting tree (based on pointers) to a flat (pointerless)
    //     representation

    // 1) initialize element info array and fill it with primitive information
    std::vector<ElementInfo> element_info;
    for (u64 i = 0; i < elements_.size(); ++i)
      element_info.emplace_back(i, elements_[i].bounds());

    // 2) build BVH tree for elements_ using element_info
    // Reserve memory to store all nodes
    MemoryArena arena(1024 * 1024);
    // total number of nodes in the tree
    i32 total_nodes = 0;
    // The build process generates an array primitives ordered so that the primitives
    // in leaf nodes occupy contiguous ranges in the array. This array is then swapped
    // with the original array later
    std::vector<T> ordered_elements;
    // Tree root pointer
    BuildNode *root{nullptr};
    // The HLBVH takes a different approach while all other algorithms happen in the
    // recursive traversal
    if (split_method_ == BVHSplitMethod::HLBVH)
      root = HLBVHBuild(arena, element_info, &total_nodes, ordered_elements);
    else
      root = recursiveBuild(arena, element_info, 0, elements_.size(), &total_nodes,
                            ordered_elements);
    // Swap the original element array by the resulting ordered array to increase cache hit
    elements_.swap(ordered_elements);
    // 3) compute representation of depth-first traversal of BVH tree
    nodes_ = allocAligned<LinearNode>(total_nodes);
    i32 offset = 0;
    flattenBVHTree(root, &offset);
  }
  ~BVH() {
    if (nodes_)
      freeAligned(nodes_);

  }
  // ***********************************************************************
  //                           PUBLIC METHODS
  // ***********************************************************************
  /// Traverses through BVH tree based on a given predicate test. If the predicate
  /// test fails the traversal stops the recursion on that branch, otherwise both
  /// children are added to the stack
  /// \param predicate the predicate must return true if traversal must
  ///                  continue, false otherwise
  /// \param process_leaf this callback is called for each leaf node that is
  ///                     visited during traversal
  void traverse(const std::function<bool(const bbox3 &)> &predicate,
                const std::function<void(const LinearNode &)> &process_leaf) const {
    i32 to_visit_offset = 0;
    i32 current_node_index = 0;
    i32 nodes_to_visit[64];
    while (true) {
      const LinearNode *node = &nodes_[current_node_index];
      if (predicate(node->bounds)) {
        if (node->n_elements > 0) {
          process_leaf(*node);
          if (to_visit_offset == 0)
            break;
          current_node_index = nodes_to_visit[--to_visit_offset];
        } else {
          nodes_to_visit[to_visit_offset++] = current_node_index + 1;
          current_node_index = node->second_child_offset;
        }
      } else {
        if (to_visit_offset == 0)
          break;
        current_node_index = nodes_to_visit[--to_visit_offset];
      }
    }
  }
  ///
  /// \param ray
  /// \param predicate
  /// \return
  std::optional<T> intersect(const ray3 &ray,
                             const std::function<std::optional<f32>(const T &)> &predicate) {
    ponos::vec3 inv_dir = {1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z};
    int dir_is_neg[3] = {inv_dir.x < 0, inv_dir.y < 0, inv_dir.z < 0};
    // follow ray through BVH nodes to find primitive intersections
    i32 to_visit_offset = 0;
    i32 current_node_index = 0;
    i32 nodes_to_visit[64];
    f32 min_t = Constants::greatest<f32>();
    std::optional<T> id;
    while (true) {
      const LinearNode *node = &nodes_[current_node_index];
      if (GeometricPredicates::intersect(node->bounds, ray, inv_dir, dir_is_neg)) {
        if (node->n_elements > 0) {
          // intersect ray with primitives in leaf BVH node
          for (u32 i = 0; i < node->n_elements; ++i) {
            auto t = predicate(elements_[node->element_offset + i]);
            if (t.has_value() && t.value() < min_t) {
              min_t = t.value();
              id = elements_[node->element_offset + i];
            }
          }
          if (to_visit_offset == 0)
            break;
          current_node_index = nodes_to_visit[--to_visit_offset];
        } else {
          if (dir_is_neg[node->axis]) {
            nodes_to_visit[to_visit_offset++] = current_node_index + 1;
            current_node_index = node->second_child_offset;
          } else {
            nodes_to_visit[to_visit_offset++] = node->second_child_offset;
            current_node_index = current_node_index + 1;
          }
        }
      } else {
        if (to_visit_offset == 0)
          break;
        current_node_index = nodes_to_visit[--to_visit_offset];
      }
    }
    return id;
  }

protected:
  /// Builds a BVH for the subset [start,end) of primitives in the **element_info** array.
  /// If the number of primitives is small enough the recursion stops and a leaf node is
  /// returned, otherwise this method partitions the elements of the array in the ranges
  /// [start,mid) and [mid,end) depending on the split algorithm.
  /// \param arena memory for new nodes
  /// \param element_info array of primitives
  /// \param start index of the first element included in the range
  /// \param end index of the first element not included in the range
  /// \param total_nodes tracks the total number of nodes that have been created
  /// \param ordered_elements used to store primitives as primitives are stored in leaf nodes
  /// \return A BVH for the subset of primitives passed in the parameters
  BuildNode *recursiveBuild(MemoryArena &arena,
                            std::vector<ElementInfo> &element_info, i32 start,
                            i32 end, i32 *total_nodes,
                            std::vector<T> &ordered_elements) {
    // Allocate new node and count it
    BuildNode *node = arena.alloc<BuildNode>();
    (*total_nodes)++;
    // compute bounds of all elements in this BVH node
    bbox3f bounds;
    for (int i = start; i < end; ++i)
      bounds = make_union(bounds, element_info[i].bounds);
    // compute the number of primitives to classify this node
    int n_elements = end - start;
    if (n_elements == 1) {
      // create leaf BuildNode
      int first_element_offset = static_cast<int>(ordered_elements.size());
      // the leaf node contains the range [start,end) of primitives
      for (int i = start; i < end; ++i)
        ordered_elements.push_back(elements_[element_info[i].element_number]);
      // initiate the leaf node and finish recursion
      node->initLeaf(first_element_offset, n_elements, bounds);
      return node;
    } else {
      // This is an interior node and the collection of primitives must be partitioned
      // between the two children subtrees. Here, we consider to partition along the coordinate
      // which has the largest extent when projecting the bounding box centroids to it.
      // 1) compute bounding box centroids to each axis and choose the one with the large extent
      // 2) partition primitives based on the split method

      // 1) compute bound of element centroids, choose split dimension dim
      bbox3f centroid_bounds;
      for (int i = start; i < end; ++i)
        centroid_bounds = make_union(centroid_bounds, element_info[i].centroid);
      int dim = centroid_bounds.maxExtent();

      // 2) partition elements into two sets and build children
      int mid = (start + end) / 2;
      // Check for degenerate case first
      if (centroid_bounds.upper[dim] == centroid_bounds.lower[dim]) {
        // In the edge case where the extent is zero, the node becomes a leaf node and the
        // recursion stops
        int first_element_offset = static_cast<int>(ordered_elements.size());
        for (int i = start; i < end; ++i)
          ordered_elements.push_back(elements_[element_info[i].element_number]);
        node->initLeaf(first_element_offset, n_elements, bounds);
        return node;
      }
      // partition elements_ based on split method
      switch (split_method_) {
      case BVHSplitMethod::SAH: {
        // Partition primitives by estimating the total cost of ray intersection, primitive
        // intersection and tree traversal. A choice of cost function, for example, could be
        // c(A, B) = t_trav + pA \sum_a t(a) + pB \sum_b t(b)
        // where t_trav it the time it takes to traverse the interior node and determine which
        // of the children the ray passes through, pA and pB are the probabilities that the
        // ray passes through each of the child nodes, a and b are the indices of primitives
        // that overlap the child regions of A and B.
        // Here we assume primitive intersection cost as 1 and traversal cost as 1/8
        if (n_elements <= 4) {
          // since SAH can be expensive on construction, the EquallyCounts approach is chosen
          // for small sets of primitives instead
          mid = (start + end) / 2;
          std::nth_element(&element_info[start], &element_info[mid],
                           &element_info[end - 1] + 1,
                           [dim](const ElementInfo &a, const ElementInfo &b) {
                             return a.centroid[dim] < b.centroid[dim];
                           });
        } else {
          // A minimal SAH cost estimate is found by considering a number of candidate partitions.
          // Instead of considering all 2n possible partitions, we divide the range along the axis
          // into a small number of buckets of equal extent and then only consider partitions at
          // bucket boundaries.
          // The sequence is:
          // 1) Generate buckets
          // 2) Distribute primitives between buckets
          // 3) Compute the cost function for each bucket boundary
          // 4) Get the partition with the minimum cost
          // 5) Create leaf or split based on minimum cost value

          // 1) allocate buckets
          constexpr int n_buckets = 12;
          struct BucketInfo {
            int count = 0;
            bbox3 bounds;
          };
          BucketInfo buckets[n_buckets];
          // 2) initialize buckets
          //    Count the number of primitives that their centroids fall into each bucket and their
          //    combined bounding boxes
          for (int i = start; i < end; ++i) {
            int b = // bucket index
                n_buckets * centroid_bounds.offset(element_info[i].centroid)[dim];
            if (b == n_buckets)
              b = n_buckets - 1;
            buckets[b].count++;
            buckets[b].bounds =
                make_union(buckets[b].bounds, element_info[i].bounds);
          }
          // 3) compute costs for splitting after each bucket
          real_t cost[n_buckets - 1];
          // for each bucket, compute the cost traversing all other buckets O(n2)
          // TODO: optimize to linear-time using a sweeping method
          for (int i = 0; i < n_buckets - 1; ++i) {
            bbox3 b0, b1;
            int count0 = 0, count1 = 0;
            // compute left child cost
            for (int j = 0; j <= i; ++j) {
              b0 = make_union(b0, buckets[j].bounds);
              count0 += buckets[j].count;
            }
            // compute right child cost
            for (int j = i + 1; j < n_buckets; ++j) {
              b1 = make_union(b1, buckets[j].bounds);
              count1 += buckets[j].count;
            }
            cost[i] = .125f +
                (count0 * b0.surfaceArea() + count1 * b1.surfaceArea()) /
                    bounds.surfaceArea();
          }
          // 4) find bucket to split at that minimizes SAH metric
          real_t min_cost = cost[0];
          int min_cost_split_bucket = 0;
          for (int i = 1; i < n_buckets - 1; i++)
            if (cost[i] < min_cost) {
              min_cost = cost[i];
              min_cost_split_bucket = i;
            }
          // 5) Create leaf or partition primitives into children nodes
          // since intersection cost is 1 here, the cost for a leaf is the number of primitives
          // in it
          real_t leaf_cost = n_elements;
          if (n_elements > max_elements_in_node_ || min_cost < leaf_cost) {
            // If the chosen bucket has a lower estimated cost than building a nod with the existing
            // primitives of if more than the maximum number of primitives allowed in a node is
            // present, partition based on bucket boundary
            ElementInfo *pmid = std::partition(
                &element_info[start], &element_info[end - 1] + 1,
                [=](const ElementInfo &info) {
                  int b = n_buckets * centroid_bounds.offset(info.centroid)[dim];
                  if (b == n_buckets)
                    b = n_buckets - 1;
                  return b <= min_cost_split_bucket;
                });
            mid = pmid - &element_info[0];
          } else {
            // otherwise a leaf is the best option
            int firstElementOffset = static_cast<int>(ordered_elements.size());
            for (int i = start; i < end; ++i)
              ordered_elements.push_back(elements_[element_info[i].element_number]);
            node->initLeaf(firstElementOffset, n_elements, bounds);
            return node;
          }
        }
      }
        break;
      case BVHSplitMethod::Middle: {
        // Computes the midpoint of the primitive's centroids along the splitting axis.
        // The primitives are then classified into two sets depending on
        // whether their centroids are above or below the midpoint.
        // 1) compute middle point based on centroids extent
        f32 pmid = (centroid_bounds.lower[dim] + centroid_bounds.upper[dim]) / 2;
        // 2) partition elements based on middle point position
        // std::partition tests all elements in an array against a predicate test. Elements
        // that pass in the test are put in the begin of the array, while elements that fail
        // are tossed to the end. std::partition then returns the addressOf of the first element
        // that does not pass in the predicate.
        ElementInfo *mid_ptr =
            std::partition(&element_info[start], &element_info[end - 1] + 1,
                           [dim, pmid](const ElementInfo &pi) {
                             return pi.centroid[dim] < pmid;
                           });
        // convert addressOf to array index
        mid = mid_ptr - &element_info[0];
        // we are done only if we could successfully separate elements, in the edge case
        // where we end up with a single subset of elements the EqualCounts algorithm is
        // applied
        if (mid != start && mid != end)
          break;
      }
      case BVHSplitMethod::EqualCounts: {
        // Partition the primitives into two equal-sized subsets such that the first half of
        // n of them are the n / 2 with the smallest centroid coordinate values along the chosen
        // axis, and the second half are the ones with the largest centroid coordinate values.
        mid = (start + end) / 2;
        // The std::nth_element function sorts the array so that the element at the middle
        // pointer is the one that would be there if the array was fully sorted, and such
        // that all of the elements after it compare to greater than it. O(n)
        std::nth_element(&element_info[start], &element_info[mid],
                         &element_info[end - 1] + 1,
                         [dim](const ElementInfo &a, const ElementInfo &b) {
                           return a.centroid[dim] < b.centroid[dim];
                         });
      }
        break;
      }
      // recurse over children
      node->initInterior(dim,
                         recursiveBuild(arena, element_info, start, mid,
                                        total_nodes, ordered_elements),
                         recursiveBuild(arena, element_info, mid, end,
                                        total_nodes, ordered_elements));
    }
    return node;
  }
  /// Morton-curve based clustering is used to first build trees for the lower levels of the
  /// hierarchy and the top levels of the tree are then created using the SAH
  /// \param arena memory for new nodes
  /// \param element_info array of primitives
  /// \param total_nodes tracks the total number of nodes that have been created
  /// \param ordered_elements used to store primitives as primitives are stored in leaf nodes
  /// \return A BVH for the primitives passed in the parameters
  BuildNode *HLBVHBuild(MemoryArena &arena,
                        std::vector<ElementInfo> &element_info, i32 *total_nodes,
                        std::vector<T> &ordered_elements) {
    /// Morton-curve based clustering is used to first build trees for the lower levels of the
    /// hierarchy (called treelets here) and the top levels of the tree are then created using
    /// the SAH
    /// The construction goes as follows:
    ///     1) Compute bounds of all primitive centroids
    ///     2) Compute Morton indices for all primitives
    ///     3) Sort primitives based on Morton indices
    ///     4) Create LBVH treelets
    ///     5) Create and return SAH BVH from LBVH treelets

    /// 1) compute bbox of all element centroids
    bbox3f bounds;
    for (const ElementInfo &ei : element_info)
      bounds = make_union(bounds, ei.centroid);
    // 2) compute morton indices of primitives
    std::vector<MortonElement> morton_elements(element_info.size());
    Parallel::loop([&](int i) {
                     constexpr int morton_bits = 10;
                     constexpr int morton_scale = 1u << morton_bits;
                     morton_elements[i].element_index = element_info[i].element_number;
                     vec3 centroid_offset = bounds.offset(element_info[i].centroid) *
                         static_cast<real_t>(morton_scale);
                     morton_elements[i].morton_code = encodeMortonCode(
                         centroid_offset.x, centroid_offset.y, centroid_offset.z);
                   },
                   element_info.size(), 512);
    // 3) radix sort element morton indices
    radixSort(&morton_elements);
    // 4) create LBVH treelets at bottom of BVH
    //  find interval of elements_ for each treelet
    std::vector<LBVHTreelet> treelets_to_build;
    // iterate over each morton element and add it to the respective treelet
    // Primitives are clustered based on the 12 most significant bits (of the 32 bits
    // that compose the morton code), giving a total of 4096 grid cells
    // Since the morton elements array is already sorted and most of these cells is
    // empty, we do not create this many treelets and do not need to sort later.
    for (int start = 0, end = 1; end <= (int) morton_elements.size(); ++end) {
      u32 mask = 0b00111111111111000000000000000000;
      if ((end == (int) morton_elements.size()) ||
          ((morton_elements[start].morton_code & mask) !=
              (morton_elements[end].morton_code & mask))) {
        // add entry to treelets_to_build for this treelet
        int n_elements = end - start;
        int max_BVH_nodes = 2 * n_elements;
        // pre-allocate nodes for the BVH (the number of nodes is bounded by twice
        // the number of elements)
        BuildNode *nodes = arena.alloc<BuildNode>(max_BVH_nodes, false);
        treelets_to_build.push_back({start, n_elements, nodes});
        start = end;
      }
    }
    //  create LBVHs for treelets in parallel
    std::atomic<int> atomic_total(0), ordered_elements_offset(0);
    ordered_elements.resize(elements_.size());
    Parallel::loop([&](int i) {
                     // generate ith LBVH treelet
                     int nodes_created = 0;
                     const int first_bit_index = 29 - 12;
                     LBVHTreelet &tr = treelets_to_build[i];
                     tr.build_nodes =
                         emitLBVH(tr.build_nodes, element_info, &morton_elements[tr.start_index],
                                  tr.n_elements, &nodes_created, ordered_elements,
                                  &ordered_elements_offset, first_bit_index);
                     atomic_total += nodes_created;
                   },
                   treelets_to_build.size());
    *total_nodes = atomic_total;
    // 5) create and return SAH BVH from LBVH treelets
    std::vector<BuildNode *> finished_treelets;
    for (LBVHTreelet &treelet : treelets_to_build)
      finished_treelets.push_back(treelet.build_nodes);
    return buildUpperSAH(arena, finished_treelets, 0, finished_treelets.size(),
                         total_nodes);
  }
  /// Buids the treelet by recursively partitioning its primitives based on the bit_index bit
  /// \param build_nodes memory region allocated for build nodes
  /// \param element_info element info array
  /// \param morton_elements sorted morton elements array
  /// \param n_elements total number of elements
  /// \param total_nodes total number of nodes generated
  /// \param ordered_elements used to store primitives as primitives are stored in leaf nodes
  /// \param ordered_elements_offset stores the next available entry in the ordered_elements array
  /// \param bit_index bit used to split primitives
  /// \return pointer to root node
  BuildNode *
  emitLBVH(BuildNode *&build_nodes, const std::vector<ElementInfo> &element_info,
           MortonElement *morton_elements, i32 n_elements, i32 *total_nodes,
           std::vector<T> &ordered_elements, std::atomic<i32> *ordered_elements_offset,
           i32 bit_index) const {
    if (bit_index == -1 || n_elements < max_elements_in_node_) {
      // create and return leaf node of LBVH treelet
      (*total_nodes)++;
      BuildNode *node = build_nodes++;
      bbox3f bounds;
      i32 first_element_offset = ordered_elements_offset->fetch_add(n_elements);
      for (i32 i = 0; i < n_elements; ++i) {
        i32 element_index = morton_elements[i].element_index;
        ordered_elements[first_element_offset + i] = elements_[element_index];
        bounds = make_union(bounds, element_info[element_index].bounds);
      }
      node->initLeaf(first_element_offset, n_elements, bounds);
      return node;
    } else {
      i32 mask = 1 << bit_index;
      // advance to next subtree level if there's no LBVH split for this bit
      if ((morton_elements[0].morton_code & mask) ==
          (morton_elements[n_elements - 1].morton_code & mask))
        return emitLBVH(build_nodes, element_info, morton_elements, n_elements,
                        total_nodes, ordered_elements, ordered_elements_offset,
                        bit_index - 1);
      // find LBVH split point for this dimension
      i32 search_start = 0, search_end = n_elements - 1;
      while (search_start + 1 != search_end) {
        i32 mid = (search_start + search_end) / 2;
        if ((morton_elements[search_start].morton_code & mask) ==
            (morton_elements[mid].morton_code & mask))
          search_start = mid;
        else
          search_end = mid;
      }
      i32 splitOffset = search_end;
      // create and return interior LBVH node
      (*total_nodes)++;
      BuildNode *node = build_nodes++;
      BuildNode *lbvh[2] = {emitLBVH(build_nodes, element_info, morton_elements,
                                     splitOffset, total_nodes, ordered_elements,
                                     ordered_elements_offset, bit_index - 1),
                            emitLBVH(build_nodes, element_info, morton_elements,
                                     n_elements - splitOffset, total_nodes,
                                     ordered_elements, ordered_elements_offset,
                                     bit_index - 1)};
      i32 axis = bit_index % 3;
      node->initInterior(axis, lbvh[0], lbvh[1]);
      return node;
    }
  }
  ///
  /// \param arena
  /// \param treelet_roots
  /// \param start
  /// \param end
  /// \param total_nodes
  /// \return
  BuildNode *buildUpperSAH(MemoryArena &arena,
                           std::vector<BuildNode *> &treelet_roots, i32 start,
                           i32 end, i32 *total_nodes) const {
    i32 nNodes = end - start;
    if (nNodes == 1)
      return treelet_roots[start];
    (*total_nodes)++;
    BuildNode *node = arena.alloc<BuildNode>();
    // Compute bounds of all nodes under this HLBVH node
    bbox3f bounds;
    for (i32 i = start; i < end; ++i)
      bounds = make_union(bounds, treelet_roots[i]->bounds);

    // Compute bound of HLBVH node centroids, choose split dimension _dim_
    bbox3f centroidBounds;
    for (i32 i = start; i < end; ++i) {
      point3 centroid = (treelet_roots[i]->bounds.lower +
          treelet_roots[i]->bounds.upper.asVector3()) *
          0.5f;
      centroidBounds = make_union(centroidBounds, centroid);
    }
    i32 dim = centroidBounds.maxExtent();
    // Allocate _BucketInfo_ for SAH partition buckets
    constexpr i32 n_buckets = 12;
    struct BucketInfo {
      i32 count = 0;
      bbox3f bounds;
    };
    BucketInfo buckets[n_buckets];

    // Initialize _BucketInfo_ for HLBVH SAH partition buckets
    for (i32 i = start; i < end; ++i) {
      real_t centroid = (treelet_roots[i]->bounds.lower[dim] +
          treelet_roots[i]->bounds.upper[dim]) *
          0.5f;
      i32 b =
          n_buckets * ((centroid - centroidBounds.lower[dim]) /
              (centroidBounds.upper[dim] - centroidBounds.lower[dim]));
      if (b == n_buckets)
        b = n_buckets - 1;
      buckets[b].count++;
      buckets[b].bounds =
          make_union(buckets[b].bounds, treelet_roots[i]->bounds);
    }

    // Compute costs for splitting after each bucket
    real_t cost[n_buckets - 1];
    for (i32 i = 0; i < n_buckets - 1; ++i) {
      bbox3f b0, b1;
      i32 count0 = 0, count1 = 0;
      for (i32 j = 0; j <= i; ++j) {
        b0 = make_union(b0, buckets[j].bounds);
        count0 += buckets[j].count;
      }
      for (i32 j = i + 1; j < n_buckets; ++j) {
        b1 = make_union(b1, buckets[j].bounds);
        count1 += buckets[j].count;
      }
      cost[i] =
          .125f + (count0 * b0.surfaceArea() + count1 * b1.surfaceArea()) /
              bounds.surfaceArea();
    }

    // Find bucket to split at that minimizes SAH metric
    real_t minCost = cost[0];
    i32 min_cost_split_bucket = 0;
    for (i32 i = 1; i < n_buckets - 1; ++i) {
      if (cost[i] < minCost) {
        minCost = cost[i];
        min_cost_split_bucket = i;
      }
    }

    // Split nodes and create i32erior HLBVH SAH node
    BuildNode **pmid = std::partition(
        &treelet_roots[start], &treelet_roots[end - 1] + 1,
        [=](const BuildNode *node) {
          real_t centroid =
              (node->bounds.lower[dim] + node->bounds.upper[dim]) * 0.5f;
          i32 b = n_buckets *
              ((centroid -
                  static_cast<const point3>(centroidBounds.lower)[dim]) /
                  (static_cast<const point3>(centroidBounds.upper)[dim] -
                      static_cast<const point3>(centroidBounds.lower)[dim]));
          if (b == n_buckets)
            b = n_buckets - 1;
          // CHECK_GE(b, 0);
          // CHECK_LT(b, nBuckets);
          return b <= min_cost_split_bucket;
        });
    i32 mid = pmid - &treelet_roots[0];
    // CHECK_GT(mid, start);
    // CHECK_LT(mid, end);
    node->initInterior(
        dim, this->buildUpperSAH(arena, treelet_roots, start, mid, total_nodes),
        this->buildUpperSAH(arena, treelet_roots, mid, end, total_nodes));
    return node;
  }
  /// Sort morton indices based on bit codes
  /// \param v
  void radixSort(std::vector<MortonElement> *v) {
    // Here, radix sort is used to sort binary codes by taking groups of digits at a time,
    // doing so reduces the total number of passes taken over the data.
    std::vector<MortonElement> temp_vector(v->size());
    /// number of bits processed per pass
    constexpr int bits_per_pass = 6;
    /// total number of bits
    constexpr int n_bits = 30;
    /// total number of passes
    constexpr int n_passes = n_bits / bits_per_pass;
    // perform one pass of radix sort, sorting bits_per_pass bits
    for (int pass = 0; pass < n_passes; ++pass) {
      // compute right most bit position for the current pass
      int low_bit = pass * bits_per_pass;
      // set in and out vector pointers for radix sort pass
      std::vector<MortonElement> &in = (pass & 1) ? temp_vector : *v;
      std::vector<MortonElement> &out = (pass & 1) ? *v : temp_vector;
      // if we are sorting n bits per pass, then there are 2^n buckets that each value can
      // land in. So we count the number of values that fall into each bucket:
      constexpr int n_buckets = 1 << bits_per_pass;
      int bucketCount[n_buckets] = {0};
      // we can compute the bucket index with the following mask:
      // 0000 0000 0000 0000 0000 0000 0011 1111 (since we process 6 bits per pass)
      constexpr int bit_mask = (1 << bits_per_pass) - 1;
      for (const MortonElement &me : in) {
        int bucket = (me.morton_code >> low_bit) & bit_mask;
        ++bucketCount[bucket];
      }
      // compute starting index in output array for each bucket by accumulate bucket counts
      int out_index[n_buckets];
      out_index[0] = 0;
      for (int i = 1; i < n_buckets; ++i)
        out_index[i] = out_index[i - 1] + bucketCount[i - 1];
      // store sorted values in output array
      for (const MortonElement &me : in) {
        int bucket = (me.morton_code >> low_bit) & bit_mask;
        out[out_index[bucket]++] = me;
      }
    }
    // copy final result from tempVector, if needed
    if (n_passes & 1)
      std::swap(*v, temp_vector);
  }
  /// Performs a depth-first tree traversal and stores nodes in memory in linear order
  /// \param node
  /// \param offset
  /// \return
  i32 flattenBVHTree(BuildNode *node, i32 *offset) {
    LinearNode *linear_node = &nodes_[*offset];
    linear_node->bounds = node->bounds;
    int my_offset = (*offset)++;
    if (node->n_elements > 0) {
      linear_node->element_offset = node->first_element_offset;
      linear_node->n_elements = node->n_elements;
    } else {
      // create interior flatten BVH node
      linear_node->axis = node->split_axis;
      linear_node->n_elements = 0;
      flattenBVHTree(node->children[0], offset);
      linear_node->second_child_offset = flattenBVHTree(node->children[1], offset);
    }
    return my_offset;
  }

  const i32 max_elements_in_node_;
  std::vector<T> elements_;
  const BVHSplitMethod split_method_;
  LinearNode *nodes_ = nullptr;
};

} // namespace ponos

#endif // PONOS_BVH_H
