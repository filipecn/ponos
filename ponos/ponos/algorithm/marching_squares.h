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

#ifndef PONOS_ALGORITHMS_MARCHING_SQUARES_H
#define PONOS_ALGORITHMS_MARCHING_SQUARES_H

#include <ponos/blas/field.h>
#include <ponos/geometry/bbox.h>
#include <ponos/geometry/numeric.h>
#include <ponos/structures/raw_mesh.h>
#include <ponos/structures/regular_grid.h>

namespace ponos {

template <typename T>
void marchingSquares(const FieldInterface2D<T> *field, const BBox2D &region,
                     size_t xResolution, size_t yResolution, RawMesh *rm,
                     T error = 1e-8) {
  RegularGrid2D<int> xEdges(xResolution, yResolution + 1, -1);
  RegularGrid2D<int> yEdges(xResolution + 1, yResolution, -1);
  RegularGrid2D<int> cells(xResolution, yResolution, 0);
  RegularGrid2D<T> grid;
  grid.dataPosition = GridDataPosition::VERTEX_CENTER;
  grid.accessMode = GridAccessMode::CLAMP_TO_EDGE;
  grid.set(xResolution, yResolution, region);
  grid.setDimensions(xResolution + 1, yResolution + 1);
  rm->clear();
  grid.forEach([&](T &v, int i, int j) {
    Point2 wp = grid.dataWorldPosition(i, j);
    v = field->sample(wp.x, wp.y);
  });
  //  3 _2__ 2
  //  3|____| 1
  //  0  0  1
  T zero = 0;
  ivec2 ij, D = cells.getDimensions();
  FOR_INDICES0_2D(D, ij) {
    int m = (grid(ij + ivec2(0, 0)) < zero) ? 1 : 0;
    m |= (grid(ij + ivec2(1, 0)) < zero) ? 2 : 0;
    m |= (grid(ij + ivec2(1, 1)) < zero) ? 4 : 0;
    m |= (grid(ij + ivec2(0, 1)) < zero) ? 8 : 0;
    cells(ij) = m;
    Point2 wp[2] = {grid.toWorld(Point2(ij[0], ij[1])),
                    grid.toWorld(Point2(ij[0] + 1, ij[1] + 1))};
    for (int i = 0; i < 2; i++) {
      if (xEdges(ij + ivec2(0, i)) < 0 &&
          (((m >> (i * 2)) & 1) ^ ((m >> (i * 2 + 1)) & 1))) {
        std::function<float(float)> f = [&](float c) -> float {
          return field->sample(c, wp[i].y);
        };
        float x = bisect(wp[0].x, wp[1].x, grid(ij + ivec2(0, i)),
                         grid(ij + ivec2(1, i)), error, f);
        rm->addVertex({x, wp[i].y});
        xEdges(ij + ivec2(0, i)) = (rm->vertices.size() / 2) - 1;
      }
      if (yEdges(ij + ivec2(i, 0)) < 0 &&
          (((m >> i) & 1) ^ ((m >> (3 - i) & 1)))) {
        std::function<float(float)> f = [&](float c) -> float {
          return field->sample(wp[i].x, c);
        };
        float y = bisect(wp[0].y, wp[1].y, grid(ij + ivec2(i, 0)),
                         grid(ij + ivec2(i, 1)), error, f);
        rm->addVertex({wp[i].x, y});
        yEdges(ij + ivec2(i, 0)) = (rm->vertices.size() / 2) - 1;
      }
    }
  }
  int edgeTable[16][2] = {{-1, -1},  // 0
                          {0, 3},    // 1
                          {1, 0},    // 2
                          {1, 3},    // 3
                          {2, 1},    // 4
                          {2, 3},    // 5
                          {2, 0},    // 6
                          {2, 3},    // 7
                          {3, 2},    // 8
                          {0, 2},    // 9
                          {3, 0},    // 10
                          {1, 2},    // 11
                          {3, 1},    // 12
                          {0, 1},    // 13
                          {3, 0},    // 14
                          {-1, -1}}; // 15
  FOR_INDICES0_2D(D, ij) {
    int edgePoint[4] = {xEdges(ij + ivec2(0, 0)), yEdges(ij + ivec2(1, 0)),
                        xEdges(ij + ivec2(0, 1)), yEdges(ij + ivec2(0, 0))};
    if (cells(ij) > 0 && cells(ij) < 15) {
      rm->addFace({{edgePoint[edgeTable[cells(ij)][0]], 0, 0},
                   {edgePoint[edgeTable[cells(ij)][1]], 0, 0}});
      if (cells(ij) == 5)
        rm->addFace({{edgePoint[0], 0, 0}, {edgePoint[1], 0, 0}});
      else if (cells(ij) == 10)
        rm->addFace({{edgePoint[1], 0, 0}, {edgePoint[2], 0, 0}});
    }
  }
  rm->meshDescriptor.count = rm->indices.size() / 2;
  rm->vertexDescriptor.count = rm->vertices.size() / 2;
  rm->meshDescriptor.elementSize = 2;
  rm->vertexDescriptor.elementSize = 2;
  rm->primitiveType = GeometricPrimitiveType::LINES;
  rm->splitIndexData();
  rm->buildInterleavedData();
}

} // ponos namespace

#endif // PONOS_ALGORITHMS_MARCHING_SQUARES_H
