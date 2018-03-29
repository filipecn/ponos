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

#ifndef PONOS_ALGORITHM_TRIANGULATE_H
#define PONOS_ALGORITHM_TRIANGULATE_H

#include <ponos/structures/raw_mesh.h>

namespace ponos {

struct MeshData {
  std::vector<int> vertexBoundaryMarker;
  std::vector<int> edgeBoundaryMarker;
  std::vector<std::vector<float>> vertexAttributes;
  std::vector<std::pair<float, float>> holes;
};

void triangulate(const RawMesh *input, const MeshData *data, RawMesh *output);

void tetrahedralize(const RawMesh *input, RawMesh *output);

} // ponos namespace

#endif // PONOS_ALGORITHM_TRIANGULATE_H
