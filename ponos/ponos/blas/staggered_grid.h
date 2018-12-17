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

#ifndef PONOS_BLAS_STAGGERED_GRID_H
#define PONOS_BLAS_STAGGERED_GRID_H

namespace ponos {

/** Staggered Grid structure. Stores velocities components on face centers and
 * the rest of data in cell center.
 */
template <typename GridType> class StaggeredGrid2D {
public:
  typedef GridType InternalGridType;
  /** Fit grid to bounding box. The resolution defines the resolution of the
   * data found on cell centers
   * \param w width **[in]** (number of cells)
   * \param h height **[in]** (number of cells)
   * \param b bounding box
   */
  void set(uint w, uint h, const bbox2 &b);
  /** Sample velocity from u and v components
   * \param wp **[in]** world position
   * \return velocity vector
   */
  vec2 sample(const point2 &wp) const;
  GridType u; //!< velocity x component on u faces
  GridType v; //!< velocity y component on v faces
  GridType p; //!< center data
};

#include <ponos/blas/staggered_grid.inl>

typedef StaggeredGrid2D<ScalarGrid2f> StaggeredGrid2f;

} // ponos namespace

#endif // PONOS_BLAS_STAGGERED_GRID_H
