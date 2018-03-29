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

template<typename NodeData>
Octree<NodeData>::Octree() : root_(nullptr), height_(0), count_(0) {}

template<typename NodeData>
Octree<NodeData>::Octree(const BBox &region, uint h) : root_(new Node(region)) {
  height_ = 0;
  count_ = 0;
  refine(root_, [h](Node &node) { return node.level() < h; }, nullptr);
}

template<typename NodeData>
Octree<NodeData>::Octree(const BBox &region,
                         const std::function<bool(Node & node)> &f)
    : root_(new Node(region, nullptr)) {
  height_ = 0;
  count_ = 0;
  refine(root_, f, nullptr);
}

template<typename NodeData>
Octree<NodeData>::Octree(const BBox &region, NodeData rootData,
                         const std::function<bool(Node & node)> &f,
                         std::function<void(Node & node)> sf)
    : root_(new Node(region, nullptr)) {
  height_ = 0;
  count_ = 0;
  root_->data = rootData;
  refine(root_, f, sf);
}

template<typename NodeData> Octree<NodeData>::~Octree() {
  clean(root_);
}

template<typename NodeData>
void Octree<NodeData>::traverse(const std::function<bool(Octree::Node &node)> &f) {
  traverse(root_, f);
}

template<typename NodeData>
void Octree<NodeData>::traverse(const std::function<bool(const Octree::Node &)> &f) const {
  traverse(root_, f);
}

template<typename NodeData>
void Octree<NodeData>::traverse(Node *node,
                                const std::function<bool(Node & node)> &f) {
  if (!node)
    return;
  if (!f(*node))
    return;
  for (auto c : node->children)
    traverse(c, f);
}

template<typename NodeData>
void Octree<NodeData>::traverse(
    Node *node, const std::function<bool(const Octree::Node &node)> &f) const {
  if (!node)
    return;
  if (!f(*node))
    return;
  for (auto c : node->children)
    traverse(c, f);
}

template<typename NodeData> void Octree<NodeData>::clean(Node *node) {
  if (!node)
    return;
  for (int i = 0; i < 8; i++)
    clean(node->children[i]);
  delete node;
}

template<typename NodeData>
void Octree<NodeData>::refine(Node *node,
                              const std::function<bool(Node & node)> &f,
                              std::function<void(Octree::Node &)> sf) {
  if (!node)
    return;
  node->id_ = count_++;
  height_ = std::max(height_, node->level_);
  if (f(*node)) {
    auto regions = node->region().splitBy8();
    for(int i = 0; i < 8; i++)
      node->children[i] = new Node(regions[i], node);
    if (sf)
      sf(*node);
    for (int i = 0; i < 8; i++) {
      node->children[i]->childNumber = i;
      refine(node->children[i], f, sf);
    }
  }
}
