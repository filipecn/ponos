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

#ifndef POSEIDON_ALGORITHMS_GRID_SOLVER_H
#define POSEIDON_ALGORITHMS_GRID_SOLVER_H

#include "elements/simulation_scene.h"

namespace poseidon {

enum class SimCellType { FLUID, SOLID, AIR };

/** Grid based fluid solver.
 */
class GridSolver2D {
public:
  GridSolver2D();
  virtual ~GridSolver2D();
  ponos::StaggeredGrid2f &getGrid() { return macGrid; }
  SimulationScene2D &getScene() { return scene; }
  virtual void set(uint rx, const ponos::BBox2D &b);
  virtual void step(double timeInterval);

  void markCells();
  void enforceBoundaries();

protected:
  SimulationScene2D scene;
  ponos::StaggeredGrid2f macGrid;
  ponos::RegularGrid2D<SimCellType> cell;
};

} // poseidon namespace

#endif // POSEIDON_ALGORITHMS_GRID_SOLVER_H
