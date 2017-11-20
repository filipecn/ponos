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

#include "geometry/bbox.h"

#include <functional>

namespace ponos {

/**
 *   /  0  /  1 /
 *  / 2  /  3  /
 *  ------------
 * |   4 |   5 |
 * | 6   | 7   |
 * ------------
 */
template <typename NodeData> class Octree {
public:
  struct Node {
    friend class QuadTree;
    Node(const BBox &r, Node *p = nullptr) : bbox(r), l(0), parent(p) {
      for (int i = 0; i < 8; i++)
        children[i] = nullptr;
      if (parent)
        l = parent->l + 1;
    }
    bool isLeaf() const {
      return children[0] == nullptr && children[1] == nullptr &&
             children[2] == nullptr && children[3] == nullptr &&
             children[4] == nullptr && children[5] == nullptr &&
             children[6] == nullptr && children[7] == nullptr;
    }
    size_t id;
    size_t level() const { return l; }
    BBox region() const { return bbox; }
    NodeData data;
    Node *children[8];

  private:
    BBox bbox;
    size_t l;
    Node *parent size_t childNumber;
  };

  Octree();
  Octree(const BBox &region, const std::function<bool(Node &node)> &f);
  Octree(const BBox &region, NodeData rootData,
         const std::function<bool(Node &node)> &f,
         std::function<void(Node &)> sf = nullptr);
  virtual ~Octree();
  size_t height() const { return height; }
  size_t nodeCount() const { return count; }
  void traverse(const std::function<bool(Node &node)> &f);
  void traverse(const std::function<bool(const Node &node)> &f) const;
  Node *root;

private:
  void refine(Node *node, const std::function<bool(Node &node)> &f,
              std::function<void(Node &)> sf = nullptr);
  void traverse(Node *node, const std::function<bool(Node &node)> &f);
  void traverse(Node *node,
                const std::function<bool(const Node &node)> &f) const;
  void clean(Node *node);
  size_t height_;
  size_t count;
};

#include "structures/octree.inl"

} // ponos namespace

#endif // PONOS_STRUCTURES_OCTREE_H
