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

#ifndef AERGIA_HELPERS_CARTESIAN_GRID_H
#define AERGIA_HELPERS_CARTESIAN_GRID_H

#include <ponos.h>

#include "scene/scene_object.h"
#include "utils/open_gl.h"

namespace aergia {

/* cartesian grid
 * Represents the main planes of a cartesian grid.
 */
class CartesianGrid : public SceneObject {
public:
  CartesianGrid() {}
  /* Constructor.
   * @d **[in]** delta in all axis
   * Creates a grid **[-d, d] x [-d, d] x [-d, d]**
   */
  CartesianGrid(int d);
  /* Constructor.
   * @dx **[in]** delta X
   * @dy **[in]** delta Y
   * @dz **[in]** delta Z
   * Creates a grid **[-dx, dx] x [-dy, dy] x [-dz, dz]**
   */
  CartesianGrid(int dx, int dy, int dz);
  /* set
   * @d **[in]** dimension index (x = 0, ...)
   * @a **[in]** lowest coordinate
   * @b **[in]** highest coordinate
   * Set the limits of the grid for an axis.
   *
   * **Example:** If we want a grid with **y** coordinates in **[-5,5]**, we
   *call **set(1, -5, 5)**
   */
  void setDimension(size_t d, int a, int b);
  /* @inherit */
  void draw() const override;

  ponos::Interval<int> planes[3];
};

} // aergia namespace

#endif // AERGIA_HELPERS_CARTESIAN_GRID_H
