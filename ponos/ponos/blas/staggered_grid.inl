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

#include <ponos/blas/staggered_grid.h>

template <typename GridType>
void StaggeredGrid2D<GridType>::set(uint w, uint h, const bbox2 &b) {
  p.dataPosition = GridDataPosition::CELL_CENTER;
  p.accessMode = GridAccessMode::RESTRICT;
  p.set(w, h, b);
  u.dataPosition = GridDataPosition::U_FACE_CENTER;
  u.accessMode = GridAccessMode::CLAMP_TO_EDGE;
  u.setTransform(p.toWorld);
  u.setDimensions(w + 1, h);
  v.dataPosition = GridDataPosition::V_FACE_CENTER;
  v.accessMode = GridAccessMode::CLAMP_TO_EDGE;
  v.setTransform(p.toWorld);
  v.setDimensions(w, h + 1);
}

template <typename GridType>
vec2 StaggeredGrid2D<GridType>::sample(const point2 &wp) const {
  return ponos::vec2(u.sample(wp.x, wp.y), v.sample(wp.x, wp.y));
}
