#include <ponos/geometry/queries.h>
#include <ponos/structures/bah.h>

#include <algorithm>
#include <vector>

namespace ponos {

BAH::BAH(Mesh2D *m) {
  mesh.reset(m);
  std::vector<BAHElement> buildData;
  for (size_t i = 0; i < mesh->getMesh()->meshDescriptor.count; i++)
    buildData.emplace_back(BAHElement(i, mesh->getMesh()->elementBBox(i)));
  uint32_t totalNodes = 0;
  orderedElements.reserve(mesh->getMesh()->meshDescriptor.count);
  root = recursiveBuild(buildData, 0, mesh->getMesh()->meshDescriptor.count,
                        &totalNodes, orderedElements);
  nodes.resize(totalNodes);
  for (uint32_t i = 0; i < totalNodes; i++)
    new (&nodes[i]) LinearBAHNode;
  uint32_t offset = 0;
  flattenBAHTree(root, &offset);
}

BAH::BAHNode *BAH::recursiveBuild(std::vector<BAHElement> &buildData,
                                  uint32_t start, uint32_t end,
                                  uint32_t *totalNodes,
                                  std::vector<uint32_t> &orderedElements) {
  (*totalNodes)++;
  BAHNode *node = new BAHNode();
  ponos::BBox2D bbox;
  for (uint32_t i = start; i < end; ++i)
    bbox = ponos::make_union(bbox, buildData[i].bounds);
  // compute all bounds
  uint32_t nElements = end - start;
  if (nElements == 1) {
    // create leaf node
    uint32_t firstElementOffset = orderedElements.size();
    for (uint32_t i = start; i < end; i++) {
      uint32_t elementNum = buildData[i].ind;
      orderedElements.emplace_back(elementNum);
    }
    node->initLeaf(firstElementOffset, nElements, bbox);
  } else {
    // compute bound of primitives
    ponos::BBox2D centroidBounds;
    for (uint32_t i = start; i < end; i++)
      centroidBounds = ponos::make_union(centroidBounds, buildData[i].centroid);
    int dim = centroidBounds.maxExtent();
    // partition primitives
    uint32_t mid = (start + end) / 2;
    if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
      node->initInterior(
          dim,
          recursiveBuild(buildData, start, mid, totalNodes, orderedElements),
          recursiveBuild(buildData, mid, end, totalNodes, orderedElements));
      return node;
    }
    // partition into equally sized subsets
    std::nth_element(&buildData[start], &buildData[mid],
                     &buildData[end - 1] + 1, ComparePoints(dim));
    node->initInterior(
        dim, recursiveBuild(buildData, start, mid, totalNodes, orderedElements),
        recursiveBuild(buildData, mid, end, totalNodes, orderedElements));
  }
  return node;
}

uint32_t BAH::flattenBAHTree(BAHNode *node, uint32_t *offset) {
  LinearBAHNode *linearNode = &nodes[*offset];
  linearNode->bounds = node->bounds;
  uint32_t myOffset = (*offset)++;
  if (node->nElements > 0) {
    linearNode->elementsOffset = node->firstElementOffset;
    linearNode->nElements = node->nElements;
  } else {
    linearNode->axis = node->splitAxis;
    linearNode->nElements = 0;
    flattenBAHTree(node->children[0], offset);
    linearNode->secondChildOffset = flattenBAHTree(node->children[1], offset);
  }
  return myOffset;
}

int BAH::intersect(const ponos::Ray2 &ray, float *t) {
  UNUSED_VARIABLE(t);
  if (!nodes.size())
    return false;
  ponos::Transform2D inv = ponos::inverse(mesh->getTransform());
  ponos::Ray2 r = inv(ray);
  int hit = 0;
  ponos::vec2 invDir(1.f / r.d.x, 1.f / r.d.y);
  uint32_t dirIsNeg[2] = {invDir.x < 0, invDir.y < 0};
  uint32_t todoOffset = 0, nodeNum = 0;
  uint32_t todo[64];
  while (true) {
    LinearBAHNode *node = &nodes[nodeNum];
    if (intersect(node->bounds, r, invDir, dirIsNeg)) {
      if (node->nElements > 0) {
        // intersect ray with primitives
        for (uint32_t i = 0; i < node->nElements; i++) {
          ponos::Point2 v0 =
              mesh->getMesh()
                  ->vertexElement(orderedElements[node->elementsOffset + i], 0)
                  .xy();
          ponos::Point2 v1 =
              mesh->getMesh()
                  ->vertexElement(orderedElements[node->elementsOffset + i], 1)
                  .xy();
          if (ponos::ray_segment_intersection(r, ponos::Segment2(v0, v1)))
            hit++;
        }
        if (todoOffset == 0)
          break;
        nodeNum = todo[--todoOffset];
      } else {
        if (dirIsNeg[node->axis]) {
          todo[todoOffset++] = nodeNum + 1;
          nodeNum = node->secondChildOffset;
        } else {
          todo[todoOffset++] = node->secondChildOffset;
          nodeNum++;
        }
      }
    } else {
      if (todoOffset == 0)
        break;
      nodeNum = todo[--todoOffset];
    }
  }
  return hit;
}

bool BAH::intersect(const ponos::BBox2D &bounds, const ponos::Ray2 &ray,
                    const ponos::vec2 &invDir,
                    const uint32_t dirIsNeg[2]) const {
  UNUSED_VARIABLE(invDir);
  UNUSED_VARIABLE(dirIsNeg);
  float hit1, hit2;
  return ponos::bbox_ray_intersection(bounds, ray, hit1, hit2);
}

bool BAH::isInside(const ponos::Point2 &p) {
  ponos::Ray2 r(p, ponos::vec2(1.2, 1.1));
  ponos::Ray2 r2(p, ponos::vec2(0.2, -1.1));

  return intersect(r, nullptr) % 2 && intersect(r2, nullptr) % 2;
}

} // ponos namespace"
