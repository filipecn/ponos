/*
 * Copyright (c) 2019 FilipeCN
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

#ifndef HERMES_ALGORITHMS_MARCHING_CUBES_H
#define HERMES_ALGORITHMS_MARCHING_CUBES_H

#include <hermes/numeric/grid.h>
#include <hermes/storage/cuda_memory_block.h>

namespace hermes {

namespace cuda {

class MarchingSquares {
public:
  /// \tparam L Memory location (DEVICE or HOST)
  /// \param f scalar distance field
  /// \param vertices **[out]**
  /// \param indices **[out]**
  /// \param isovalue **[in]**
  static void extractIsoline(Grid2<float> &f, array1f &vertices,
                             array1u &indices, float isovalue = 0.f);
};

class MarchingCubes {
public:
  /// \tparam L Memory location (DEVICE or HOST)
  /// \param f scalar distance field
  /// \param vertices **[out]**
  /// \param indices **[out]**
  /// \param isovalue **[in]**
  template <MemoryLocation L>
  static void extractSurface(RegularGrid3<L, float> &f,
                             CuMemoryBlock1f &vertices,
                             CuMemoryBlock1u &indices, float isovalue = 0.f,
                             CuMemoryBlock1f *normals = nullptr);
};

} // namespace cuda

} // namespace hermes

#endif
