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
    Node(const BBox2D &r, Node *p = nullptr) : bbox(r), l(0), parent(p) {
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
void buildLeafNeighbourhood(QuadTree<NodeData> *tree) {
  typedef typename QuadTree<NodeData>::Node NodeType;
  std::function<void(NodeType *, NodeType *)> horizontalProcess =
      [&](NodeType *left, NodeType *right) {
        if (!left || !right)
          return;
        if (left->isLeaf() && right->isLeaf()) {
          left->data.neighbours.emplace_back(right);
          right->data.neighbours.emplace_back(left);
        } else if (left->isLeaf()) {
          horizontalProcess(left, right->children[0]);
          horizontalProcess(left, right->children[2]);
        } else if (right->isLeaf()) {
          horizontalProcess(left->children[1], right);
          horizontalProcess(left->children[3], right);
        } else {
          horizontalProcess(left->children[1], right->children[0]);
          horizontalProcess(left->children[3], right->children[2]);
        }
      };
  std::function<void(NodeType *, NodeType *)> verticalProcess =
      [&](NodeType *top, NodeType *bottom) {
        if (!top || !bottom)
          return;
        if (top->isLeaf() && bottom->isLeaf()) {
          top->data.neighbours.emplace_back(bottom);
          bottom->data.neighbours.emplace_back(top);
        } else if (top->isLeaf()) {
          verticalProcess(top, bottom->children[0]);
          verticalProcess(top, bottom->children[1]);
        } else if (bottom->isLeaf()) {
          verticalProcess(top->children[2], bottom);
          verticalProcess(top->children[3], bottom);
        } else {
          verticalProcess(top->children[2], bottom->children[0]);
          verticalProcess(top->children[3], bottom->children[1]);
        }
      };
  std::function<void(NodeType *)> faceProcess = [&](NodeType *node) {
    if (!node)
      return;
    for (auto child : node->children)
      faceProcess(child);
    horizontalProcess(node->children[0], node->children[1]);
    horizontalProcess(node->children[2], node->children[3]);
    verticalProcess(node->children[0], node->children[2]);
    verticalProcess(node->children[1], node->children[3]);
  };
  faceProcess(tree->root);
}

template <typename NodeData>
void buildLeafPhantomNeighbours(QuadTree<NodeData> *tree) {
  typedef typename QuadTree<NodeData>::Node NodeType;
  std::function<void(NodeType *, NodeType *)> verticalProcess = [&](
      NodeType *top, NodeType *bottom) {
    if (!top && !bottom)
      return;
    if (top) {
      if (top->isLeaf()) {
        BBox2D r = top->region();
        top->data.neighbours.emplace_back(
            new NodeType(BBox2D(Point2(r.pMin.x, r.pMin.y - r.size(1)),
                                Point2(r.pMax.x, r.pMin.y))));
        top->data.neighbours[top->data.neighbours.size() - 1]->data.isPhantom =
            true;
      } else {
        verticalProcess(top->children[2], bottom);
        verticalProcess(top->children[3], bottom);
      }
    } else {
      if (bottom->isLeaf()) {
        BBox2D r = bottom->region();
        bottom->data.neighbours.emplace_back(
            new NodeType(BBox2D(Point2(r.pMin.x, r.pMax.y),
                                Point2(r.pMax.x, r.pMax.y + r.size(1)))));
        bottom->data.neighbours[bottom->data.neighbours.size() - 1]
            ->data.isPhantom = true;
      } else {
        verticalProcess(top, bottom->children[0]);
        verticalProcess(top, bottom->children[1]);
      }
    }
  };
  verticalProcess(tree->root, nullptr);
  verticalProcess(nullptr, tree->root);
  std::function<void(NodeType *, NodeType *)> horizontalProcess =
      [&](NodeType *left, NodeType *right) {
        if (!left && !right)
          return;
        if (left) {
          if (left->isLeaf()) {
            BBox2D r = left->region();
            left->data.neighbours.emplace_back(
                new NodeType(BBox2D(Point2(r.pMax.x, r.pMin.y),
                                    Point2(r.pMax.x + r.size(0), r.pMax.y))));
            left->data.neighbours[left->data.neighbours.size() - 1]
                ->data.isPhantom = true;
          } else {
            horizontalProcess(left->children[1], right);
            horizontalProcess(left->children[3], right);
          }
        } else {
          if (right->isLeaf()) {
            BBox2D r = right->region();
            right->data.neighbours.emplace_back(
                new NodeType(BBox2D(Point2(r.pMin.x - r.size(0), r.pMin.y),
                                    Point2(r.pMin.x, r.pMax.y))));
            right->data.neighbours[right->data.neighbours.size() - 1]
                ->data.isPhantom = true;
          } else {
            horizontalProcess(left, right->children[0]);
            horizontalProcess(left, right->children[2]);
          }
        }
      };
  horizontalProcess(tree->root, nullptr);
  horizontalProcess(nullptr, tree->root);
}

#include "structures/quad_tree.inl"

} // ponos namespace

#endif // PONOS_STRUCTURES_QUAD_TREE_H