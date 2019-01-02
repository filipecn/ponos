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

#ifndef HELIOS_ACCELERATORS_H_BVH_H
#define HELIOS_ACCELERATORS_H_BVH_H

#include <helios/accelerators/aggregate.h>
#include <ponos/structures/bvh.h>

namespace helios {

class HBVH : public Aggregate, ponos::BVH<Primitive> {
public:
  /// \param o object smart pointer array
  /// \param maxElementsInNode maximum number of elements allowed in a node
  /// \param method split method
  HBVH(const std::vector<std::shared_ptr<Primitive>> &o, int maxElementsInNode,
       ponos::BVHSplitMethod method);

  bool intersect(const HRay &ray, SurfaceInteraction *isect) const override;
  bool intersectP(const HRay &r) const override;
};

} // namespace helios

#endif // HELIOS_ACCELERATORS_H_BVH_H