/*
 * Copyright (c) 2017 FilipeCN
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

#ifndef PONOS_STRUCTURES_OCTREE_H
#define PONOS_STRUCTURES_OCTREE_H

#include <ponos/geometry/bbox.h>

#include <functional>

namespace ponos {

/**
 * y
 * |_ x
 * z
 *   /  2  /  3 /
 *  / 6  /  7  /
 *  ------------
 * |   0 |   1 |
 * | 4   | 5   |
 * ------------
 */
/// Represents an Octree
/// \tparam NodeData Data Type stored in each node
template<typename NodeData> class Octree {
public:
  struct Node {
    friend class Octree;
    /// Constructor
    /// \param r root's region on space
    /// \param p [opitional] a parent node, if so, the new node becomes its child
    explicit Node(const bbox3 &r, Node *p = nullptr) : bbox_(r), level_(0), parent(p) {
      for (int i = 0; i < 8; i++)
        children[i] = nullptr;
      if (parent)
        level_ = parent->level_ + 1;
    }
    /// checks if node has any child
    /// \return **true** if leaf
    bool isLeaf() const {
      return children[0] == nullptr && children[1] == nullptr &&
          children[2] == nullptr && children[3] == nullptr &&
          children[4] == nullptr && children[5] == nullptr &&
          children[6] == nullptr && children[7] == nullptr;
    }
    /// node id on the tree
    /// \return id
    uint id() const { return id_; }
    /// node level on the tree
    /// \return level
    uint level() const { return level_; }
    /// region in space the node occupies
    /// \return a bounding box represent the node's region
    bbox3 region() const { return bbox_; }
    NodeData data;
    Node *children[8];

  private:
    uint id_;         ///< node's unique id on the tree
    bbox3 bbox_;       ///< region in space this node occupies
    uint level_;      ///< level on the tree of this node
    Node *parent;     ///< pointer to parent node
    uint childNumber; ///< which child is this
  };
  /// Constructor
  Octree();
  /// Builds full octree
  /// \param region root's region on space
  /// \param height tree's height
  Octree(const bbox3 &region, uint height);
  /// Constructor
  /// \param region  root's region on space
  /// \param f refinement criteria
  Octree(const bbox3 &region, const std::function<bool(Node &node)> &f);
  /// Constructor
  /// \param region root's region on space
  /// \param rootData root's data
  /// \param f refinement criteria
  /// \param sf **[opitional]** called right after children are created (pre-order traversal)
  Octree(const bbox3 &region, NodeData rootData,
         const std::function<bool(Node &node)> &f,
         std::function<void(Node &)> sf = nullptr);
  virtual ~Octree();
  /// \return tree's height
  uint height() const { return height_; }
  /// \return total number of nodes
  uint nodeCount() const { return count_; }
  /// pre-order traversal
  /// \param f if returns false, stops recursion
  void traverse(const std::function<bool(Node &node)> &f);
  /// const pre-order traversal
  /// \param f if returns false, stops recursion
  void traverse(const std::function<bool(const Node &node)> &f) const;

protected:
  /// refine tree
  /// \param node starting node
  /// \param f refinement criteria
  /// \param sf **[opitional]** called right after children are created (pre-order traversal)
  void refine(Node *node, const std::function<bool(Node &node)> &f,
              std::function<void(Node &)> sf = nullptr);
  /// pre-order traversal
  /// \param node starting node
  /// \param f if returns false, stops recursion
  void traverse(Node *node, const std::function<bool(Node &node)> &f);
  /// const pre-order traversal
  /// \param node starting node
  /// \param f if returns false, stops recursion
  void traverse(Node *node,
                const std::function<bool(const Node &node)> &f) const;
  /// free memory
  void clean(Node *node);
  Node *root_;    ///< tree's root
  uint height_;   ///< current height
  uint count_;    ///< current node count
};

#include "ponos/structures/octree.inl"

} // ponos namespace

#endif // PONOS_STRUCTURES_OCTREE_H
