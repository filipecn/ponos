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

#include "structures/octree.h"

template <typename NodeData> Octree<NodeData>::Octree() : root(nullptr) {
  height_ = 0;
  count = 0;
}

template <typename NodeData>
Octree<NodeData>::Octree(const BBox &region,
                         const std::function<bool(Node &node)> &f)
    : root(new Node(region, nullptr)) {
  height_ = 0;
  count = 0;
  refine(root, f, nullptr);
}

template <typename NodeData>
Octree<NodeData>::Octree(const BBox &region, NodeData rootData,
                         const std::function<bool(Node &node)> &f,
                         std::function<void(Node &)> sf)
    : root(new Node(region, nullptr)) {
  height_ = 0;
  count = 0;
  root->data = rootData;
  refine(root, f, sf);
}

template <typename NodeData> Octree<NodeData>::~Octree() {
  clean(root);
  height_ = count = 0;
}

template <typename NodeData>
void Octree<NodeData>::traverse(const std::function<bool(Node &node)> &f) {
  traverse(root, f);
}

template <typename NodeData>
void Octree<NodeData>::traverse(
    const std::function<bool(const Node &node)> &f) const {
  traverse(root, f);
}

template <typename NodeData>
void Octree<NodeData>::traverse(Node *node,
                                const std::function<bool(Node &node)> &f) {
  if (!node)
    return;
  if (!f(*node))
    return;
  for (auto c : node->children)
    traverse(c, f);
}

template <typename NodeData>
void Octree<NodeData>::traverse(
    Node *node, const std::function<bool(const Node &node)> &f) const {
  if (!node)
    return;
  if (!f(*node))
    return;
  for (auto c : node->children)
    traverse(c, f);
}

template <typename NodeData> void Octree<NodeData>::clean(Node *node) {
  if (!node)
    return;
  for (int i = 0; i < 8; i++)
    clean(node->children[i]);
  delete node;
}

template <typename NodeData>
void Octree<NodeData>::refine(Node *node,
                              const std::function<bool(Node &node)> &f,
                              std::function<void(Node &)> sf) {
  if (!node)
    return;
  node->id = count++;
  height_ = std::max(height_, node->l);
  Point3 pmin = node->bbox.pMin;
  Point3 pmax = node->bbox.pMax;
  Point3 mid = node->bbox.center();
  if (f(*node)) {
    node->children[0] = new Node(
        BBox(Point3(pmin.x, mid.y, mid.z), Point3(mid.x, pmax.y, mid.z)), node);
    node->children[1] = new Node(BBox(mid, pmax), node);
    node->children[2] = new Node(BBox(pmin, mid), node);
    node->children[3] =
        new Node(BBox2D(Point2(mid.x, pmin.y), Point2(pmax.x, mid.y)), node);
    if (sf)
      sf(*node);
    for (int i = 0; i < 4; i++) {
      node->children[i]->childNumber = i;
      refine(node->children[i], f, sf);
    }
  }
}
