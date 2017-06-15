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

#ifndef PONOS_STRUCTURES_QUAD_TREE_H
#define PONOS_STRUCTURES_QUAD_TREE_H

#include "geometry/bbox.h"

#include <functional>

namespace ponos {

//  | 0 | 1 |
//  | 2 | 3 |
template <typename NodeData> class QuadTree {
public:
  struct Node {
    friend class QuadTree;
    Node(const BBox2D &r, Node *p) : bbox(r), l(0), parent(p) {
      for (int i = 0; i < 4; i++)
        children[i] = nullptr;
      if (parent)
        l = parent->l + 1;
    }
    bool isLeaf() const {
      return children[0] == nullptr && children[1] == nullptr &&
             children[2] == nullptr && children[3] == nullptr;
    }
    size_t id;
    size_t level() const { return l; }
    BBox2D region() const { return bbox; }
    NodeData data;
    Node *children[4];

  private:
    BBox2D bbox;
    size_t l;
    Node *parent;
    size_t childNumber;
  };

  QuadTree();
  QuadTree(const BBox2D &region, const std::function<bool(Node &node)> &f);
  virtual ~QuadTree();
  size_t height() const { return height; }
  size_t nodeCount() const { return count; }
  void traverse(const std::function<bool(Node &node)> &f);
  void traverse(const std::function<bool(const Node &node)> &f) const;
  Node *root;

private:
  void refine(Node *node, const std::function<bool(Node &node)> &f);
  void traverse(Node *node, const std::function<bool(Node &node)> &f);
  void traverse(Node *node,
                const std::function<bool(const Node &node)> &f) const;
  void clean(Node *node);
  size_t height_;
  size_t count;
};

template <typename NodeData> struct NeighbourQuadTreeNode {
  NeighbourQuadTreeNode() { isPhantom = false; }
  NodeData data;
  bool isPhantom;
  std::vector<typename QuadTree<NeighbourQuadTreeNode<NodeData>>::Node *>
      neighbours;
};

template <typename NodeData>
void buildLeafNeighbourhood(QuadTree<NeighbourQuadTreeNode<NodeData>> *tree) {
  typedef QuadTree<NeighbourQuadTreeNode<NodeData>>::Node NodeType;
  std::function<void(NodeType *, NodeType *)> horizontalConnect =
      [&](NodeType *left, NodeType *right) {
        if (!left && !right)
          return;
        if (left && left->isLeaf() && !right) {
          BBox2D region = left->region();
          size_t n = left->data.neighbours.size();
          left->data.neighbours.emplace_back(new NodeType(
              BBox2D(Point2(region.pMax.x, region.pMin.y),
                     Point2(region.pMax.x + region.size(0), region.pMax.y)),
              nullptr));
          left->data.neighbours[n]->data.isPhantom = true;
        } else if (left && !right) {
          horizontalConnect(left->children[1], right);
          horizontalConnect(left->children[3], right);
        } else if (!left && right && right->isLeaf()) {
          BBox2D region = right->region();
          size_t n = right->data.neighbours.size();
          right->data.neighbours.emplace_back(new NodeType(
              BBox2D(Point2(region.pMin.x - region.size(0), region.pMin.y),
                     Point2(region.pMin.x, region.pMax.y)),
              nullptr));
          right->data.neighbours[n]->data.isPhantom = true;
        } else if (!left && right) {
          horizontalConnect(left, right->children[0]);
          horizontalConnect(left, right->children[2]);
        } else if (left->isLeaf() && right->isLeaf()) {
          left->data.neighbours.emplace_back(right);
          right->data.neighbours.emplace_back(left);
        } else if (left->isLeaf()) {
          horizontalConnect(left, right->children[0]);
          horizontalConnect(left, right->children[2]);
        } else if (right->isLeaf()) {
          horizontalConnect(left->children[1], right);
          horizontalConnect(left->children[3], right);
        } else {
          horizontalConnect(left->children[1], right->children[0]);
          horizontalConnect(left->children[3], right->children[2]);
        }
        if (left && !left->isLeaf()) {
          horizontalConnect(left->children[0], left->children[1]);
          horizontalConnect(left->children[2], left->children[3]);
        }
        if (right && !right->isLeaf()) {
          horizontalConnect(right->children[0], right->children[1]);
          horizontalConnect(right->children[2], right->children[3]);
        }
      };
  std::function<void(NodeType *, NodeType *)> verticalConnect =
      [&](NodeType *top, NodeType *bottom) {
        if (!top && !bottom)
          return;
        if (top && top->isLeaf() && !bottom) {
          BBox2D region = top->region();
          size_t n = top->data.neighbours.size();
          top->data.neighbours.emplace_back(new NodeType(
              BBox2D(Point2(region.pMin.x, region.pMin.y - region.size(1)),
                     Point2(region.pMax.x, region.pMin.y)),
              nullptr));
          top->data.neighbours[n]->data.isPhantom = true;
        } else if (top && !bottom) {
          verticalConnect(top->children[2], bottom);
          verticalConnect(top->children[3], bottom);
        } else if (!top && bottom && bottom->isLeaf()) {
          BBox2D region = bottom->region();
          size_t n = bottom->data.neighbours.size();
          bottom->data.neighbours.emplace_back(new NodeType(
              BBox2D(Point2(region.pMin.x, region.pMax.y),
                     Point2(region.pMax.x, region.pMax.y + region.size(1))),
              nullptr));
          bottom->data.neighbours[n]->data.isPhantom = true;
        } else if (!top && bottom) {
          verticalConnect(top, bottom->children[0]);
          verticalConnect(top, bottom->children[1]);
        } else if (top->isLeaf() && bottom->isLeaf()) {
          top->data.neighbours.emplace_back(bottom);
          bottom->data.neighbours.emplace_back(top);
        } else if (top->isLeaf()) {
          verticalConnect(top, bottom->children[0]);
          verticalConnect(top, bottom->children[1]);
        } else if (bottom->isLeaf()) {
          verticalConnect(top->children[2], bottom);
          verticalConnect(top->children[3], bottom);
        } else {
          verticalConnect(top->children[2], bottom->children[0]);
          verticalConnect(top->children[3], bottom->children[1]);
        }
        if (top && !top->isLeaf()) {
          verticalConnect(top->children[0], top->children[2]);
          verticalConnect(top->children[1], top->children[3]);
        }
        if (bottom && !bottom->isLeaf()) {
          verticalConnect(bottom->children[0], bottom->children[2]);
          verticalConnect(bottom->children[1], bottom->children[3]);
        }
      };
  horizontalConnect(tree->root, nullptr);
  horizontalConnect(nullptr, tree->root);
  verticalConnect(tree->root, nullptr);
  verticalConnect(nullptr, tree->root);
}

#include "structures/quad_tree.inl"

} // ponos namespace

#endif // PONOS_STRUCTURES_QUAD_TREE_H
