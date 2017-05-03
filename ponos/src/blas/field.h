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

namespace ponos {

class FieldInterface {
public:
  FieldInterface() {}
  virtual ~FieldInterface() {}
};

class ScalarField : public FieldInterface {
  virtual double sample(const Point3 &p) const = 0;
  virtual vec3 gradient(const Point3 &p) const = 0;
  virtual double laplacian(const Point3 &p) const = 0;
};

class VectorField : public FieldInterface {
  virtual vec3 sample(const Point3 &p) const = 0;
  virtual double divergence(const Point3 &p) const = 0;
  virtual vec3 curl(const Point3 &p) const = 0;
};

} // ponos namespace

#endif // PONOS_BLAS_FIELD_H
