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

#ifndef PONOS_BLAS_FIELD_GRID_H
#define PONOS_BLAS_FIELD_GRID_H

#include "blas/field.h"
#include "common/macros.h"
#include "log/debug.h"
#include "blas/c_regular_grid.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace ponos {

/**
 */
template <typename T>
class ScalarGrid2D : public CRegularGrid2D<T>, virtual public ScalarField2D<T> {
public:
  ScalarGrid2D();
  ScalarGrid2D(uint w, uint h);
  ScalarGrid2D(uint w, uint h, const BBox2D &b);
  virtual ~ScalarGrid2D() {}
  Vector<T, 2> gradient(int x, int y) const;
  Vector<T, 2> gradient(float x, float y) const override;
  T laplacian(float x, float y) const override;
  T sample(float x, float y) const override;
};

#include "field_grid.inl"

typedef ScalarGrid2D<float> ScalarGrid2f;

} // ponos namespace

#endif // PONOS_BLAS_FIELD_GRID_H
