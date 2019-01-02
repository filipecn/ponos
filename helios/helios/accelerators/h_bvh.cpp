/*
 * Copyright (c) 2018 FilipeCN
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

#include "h_bvh.h"
#include <helios/geometry/queries.h>
#include <ponos/geometry/point.h>

using namespace ponos;

namespace helios {

HBVH::HBVH(const std::vector<std::shared_ptr<Primitive>> &o,
           int maxElementsInNode, BVHSplitMethod method)
    : BVH<Primitive>(o, maxElementsInNode, method) {}

bool HBVH::intersect(const HRay &ray, SurfaceInteraction *isect) const {
  bool hit = false;
  vec3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
  int dirIsNeg[3] = {invDir.x < 0, invDir.y < 0, invDir.z < 0};
  // follow ray through BVH nodes to find primitive interesctions
  int toVisitOffset = 0, currentNodeIndex = 0;
  int nodesToVisit[64];
  while (true) {
    const BVH<Primitive>::LinearNode *node = &this->nodes[currentNodeIndex];
    // check ray against BVH node
    if (Queries::intersectP(node->bounds, ray, invDir, dirIsNeg)) {
      if (node->nElements > 0) {
        // intersect ray with primitives in leaf BVH node
        for (int i = 0; i < node->nElements; ++i)
          if (elements[node->elementOffset + i]->intersect(ray, isect))
            hit = true;
        if (toVisitOffset == 0)
          break;
        currentNodeIndex = nodesToVisit[--toVisitOffset];
      } else {
        // put fab BVH node on noesToVisit stack, advance to near node
        if (dirIsNeg[node->axis]) {
          nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
          currentNodeIndex = node->secondChildOffset;
        } else {
          nodesToVisit[toVisitOffset++] = node->secondChildOffset;
          currentNodeIndex = currentNodeIndex + 1;
        }
      }
    } else {
      if (toVisitOffset == 0)
        break;
      currentNodeIndex = nodesToVisit[--toVisitOffset];
    }
  }
  return hit;
}

bool HBVH::intersectP(const HRay &r) const {}

} // namespace helios