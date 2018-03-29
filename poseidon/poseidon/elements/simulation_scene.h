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

#ifndef POSEIDON_ELEMETNS_SIMULATION_SCENE_H
#define POSEIDON_ELEMETNS_SIMULATION_SCENE_H

#include <ponos.h>
#include "elements/collider.h"

namespace poseidon {

/** Describes the objects in scene that will interact with the simulation.
 */
class SimulationScene2D {
public:
  void set(size_t w, size_t h, const ponos::BBox2D &b);
  void addCollider(const Collider2D &c);
  const ponos::ScalarGrid2f *getSDF() const { return &csdf; }

private:
  std::vector<Collider2D> colliders;
  ponos::ScalarGrid2f csdf;
};
} // poseidon namespace

#endif // POSEIDON_ELEMETNS_SIMULATION_SCENE_H
