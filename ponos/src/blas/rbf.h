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

#ifndef PONOS_BLAS_RBF_H
#define PONOS_BLAS_RBF_H

#include "blas/field.h"
#include "blas/symmetric_matrix.h"
#include "geometry/vector.h"
#include <functional>

namespace ponos {

template <typename T> class Kernel {
public:
  virtual T operator()(T r) const = 0;
  virtual T D(T r) const { return r; }
  virtual T D2(T r) const { return r; }
  virtual T L(T r) const { return r; }
};

template <typename T> class InverseMultiquadricKernel : public Kernel<T> {
public:
  InverseMultiquadricKernel(T epsilon) : e(epsilon) {}
  virtual T operator()(T r) const override {
    return 1 / std::sqrt(1 + SQR(e * r));
  }
  virtual T D(T r) const override { return -SQR(e) * r * pow((*this)(r), 3); }
  virtual T D2(T r) const override {
    return SQR(e) * (2.0 * SQR(e * r) - 1.0) * pow((*this)(r), 5);
  }
  virtual T L(T r) const override {
    return D2(r) - SQR(e) * pow((*this)(r), 3);
  }
  T e;
};

template <typename T> class GaussianKernel : public Kernel<T> {
public:
  GaussianKernel(T ep) : e(ep) {}
  virtual T operator()(T r) const override { return exp(-SQR(e * r)); }
  virtual T D(T r) const override { return -2.0 * SQR(e) * r * (*this)(r); }
  virtual T D2(T r) const override {
    return (4.0 * SQR(e) * SQR(e) * SQR(r) - 2.0 * SQR(e)) * (*this)(r);
  }
  virtual T L(T r) const override { return D2(r) - 2.0 * SQR(e) * (*this)(r); }
  T e;
};

template <typename T> class PowKernel : public Kernel<T> {
public:
  PowKernel(T _e) : e(_e) {}
  virtual T operator()(T r) const override { return pow(r, e); }

private:
  T e;
};

template <typename T, typename P>
class RBF2D : virtual public ScalarField2D<float> {
public:
  RBF2D() {}
  RBF2D(const std::vector<P> &p, Kernel<T> *k,
        const std::function<T(const P &, const P &)> &d2f =
            [](const P &a, const P &b) -> T { return 0.0; })
      : phi(k), points(p) {
    d2Function = d2f;
  }
  virtual T sample(float x, float y) const override {
    float sum = 0.f;
    P p(x, y);
    for (size_t i = 0; i < points.size(); i++)
      sum += weights[i] * (*phi)(d2Function(points[i], p));
    return sum;
  }
  virtual Vector<T, 2> gradient(float x, float y) const override {
    return Vector<T, 2>();
  }
  virtual T laplacian(float x, float y) const override { return 0.0; }
  void setWeights(const std::vector<T> &w) {
    weights.clear();
    std::copy(w.begin(), w.end(), std::back_inserter(weights));
  }
  const std::vector<P> &getPoints() const { return points; }
  const std::vector<T> &getWeights() const { return weights; }
  Kernel<T> *phi;
  std::function<T(const P &, const P &)> d2Function;

private:
  const std::vector<P> &points;
  std::vector<T> weights;
};
} // ponos namespace

#endif // PONOS_BLAS_RBF_H
