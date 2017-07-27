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

#ifndef PONOS_BLAS_FIELD_H
#define PONOS_BLAS_FIELD_H

#include "geometry/point.h"

namespace ponos {

template <class T> class FieldInterface2D {
public:
  FieldInterface2D() {}
  virtual ~FieldInterface2D() {}
  virtual T sample(float x, float y) const = 0;
};

template <class T> class ScalarField2D : virtual public FieldInterface2D<T> {
public:
  virtual Vector<T, 2> gradient(float x, float y) const = 0;
  virtual T laplacian(float x, float y) const = 0;
};

template <typename T>
class VectorField2D : virtual public FieldInterface2D<Vector<T, 2>> {
public:
  virtual T divergence(float x, float y) const = 0;
  virtual Vector<T, 2> curl(float x, float y) const = 0;
};

} // ponos namespace

#endif // PONOS_BLAS_FIELD_H
