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

#include "algorithms/grid_solver.h"

namespace poseidon {

GridSolver2D::GridSolver2D() {}

GridSolver2D::~GridSolver2D() {}

void GridSolver2D::set(uint rx, const ponos::BBox2D &b) {
  float dx_ = b.size(0) / static_cast<float>(rx);
  uint ry = b.size(1) / dx_;
  macGrid.set(rx, ry, ponos::BBox2D(b.pMin, ponos::Point2(rx * dx_, ry * dx_)));
  scene.set(rx, ry, ponos::BBox2D(b.pMin, ponos::Point2(rx * dx_, ry * dx_)));
  cell.accessMode = ponos::GridAccessMode::BORDER;
  cell.border = SimCellType::SOLID;
}

void GridSolver2D::step(double timeInterval) {}

void GridSolver2D::markCells() {
  cell.forEach([&](SimCellType &t, int i, int j) {
    ponos::Point2 wp = macGrid.p.dataWorldPosition(i, j);
    if (scene.getSDF()->sample(wp.x, wp.y) < 0.0)
      t = SimCellType::SOLID;
    else
      t = SimCellType::AIR;
  });
}

void GridSolver2D::enforceBoundaries() {
  /* cell.forEach([&](SimCellType &t, int i, int j) {
     if ((t == SimCellType::SOLID && cell(i - 1, j) != SimCellType::SOLID) ||
         (t != SimCellType::SOLID && cell(i - 1, j) == SimCellType::SOLID))
       macGrid.u(i, j) = 0.f;
     if ((t == SimCellType::SOLID && cell(i + 1, j) != SimCellType::SOLID) ||
         (t != SimCellType::SOLID && cell(i + 1, j) == SimCellType::SOLID))
       macGrid.u(i + 1, j) = 0.f;
   });*/
  macGrid.u.forEach([&](float &vx, int i, int j) {
    ponos::Point2 wp = macGrid.u.dataWorldPosition(i, j);
    std::cout << "entering point " << i << " " << j << std::endl;
    std::cout << wp;
    std::cout << "sampled distance " << scene.getSDF()->sample(wp.x, wp.y)
              << std::endl;
    if (scene.getSDF()->sample(wp.x, wp.y) <= 0.0) {
      ponos::vec2f cv;
      ponos::vec2 v = macGrid.sample(wp);
      ponos::vec2f g = scene.getSDF()->gradient(wp.x, wp.y);
      std::cout << "sampled velocity " << v;
      if (g.length2() > 0.0f) {
        ponos::vec2f gn = g.normalized();
        ponos::Normal n(gn[0], gn[1], 0.f);
        std::cout << n;
        ponos::vec2f velp(v.x - cv[0], v.y - cv[1]);
        ponos::vec3 pvelp = n.project(ponos::vec3(velp[0], velp[1], 0.f));
        vx = pvelp.x + cv[0];
      } else {
        vx = cv[0];
      }
    }
  });
}

} // poseidon namespace
