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

#include "queries.h"

using namespace ponos;

namespace helios {

bool Queries::intersectP(const ponos::bbox3f &bounds, const HRay &ray,
                         const ponos::vec3f &invDir, const int dirIsNeg[3]) {
  // check for ray intersection against x and y slabs
  real_t tMin = (bounds[dirIsNeg[0]].x - ray.o.x) * invDir.x;
  real_t tMax = (bounds[1 - dirIsNeg[0]].x - ray.o.x) * invDir.x;
  real_t tyMin = (bounds[dirIsNeg[1]].y - ray.o.y) * invDir.y;
  real_t tyMax = (bounds[1 - dirIsNeg[1]].y - ray.o.y) * invDir.y;
  // update tMax and tyMax to ensure robust bounds intersection
  if (tMin > tyMax || tyMin > tMax)
    return false;
  if (tyMin > tMin)
    tMin = tyMin;
  if (tyMax < tMax)
    tMax = tyMax;
  // check for ray intersection against z slab
  real_t tzMin = (bounds[dirIsNeg[2]].z - ray.o.z) * invDir.z;
  real_t tzMax = (bounds[1 - dirIsNeg[2]].z - ray.o.z) * invDir.z;
  if (tMin > tzMax || tzMin > tMax)
    return false;
  if (tzMin > tMin)
    tMin = tzMin;
  if (tzMax < tMax)
    tMax = tzMax;
  return (tMin < ray.max_t) && (tMax > 0);
}

} // namespace helios