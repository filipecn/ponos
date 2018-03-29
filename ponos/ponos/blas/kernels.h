/*
 * Copyright (c) 2018 FilipeCN
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

#ifndef PONOS_BLAS_KERNELS_H
#define PONOS_BLAS_KERNELS_H

namespace ponos {

template<typename T> class Kernel {
public:
  virtual T operator()(T r) const = 0;
  virtual T D(T r) const { return r; }
  virtual T D2(T r) const { return r; }
  virtual T L(T r) const { return r; }
};

template<typename T> class Wendland33Kernel : public Kernel<T> {
public:
  Wendland33Kernel(T _e) : e(_e) {}
  virtual T operator()(T r) const override {
    return pow(std::max(0.f, 1.f - e * r), 8) *
        (32.f * CUBE(e * r) + 25.f * SQR(e * r) + 8.f * e * r + 1.f);
  }
  virtual T D(T r) const override {
    return -22.f * SQR(e) * pow(std::max(0.f, 1.f - e * r), 7) +
        (16.f * SQR(e * r) + 7.f * e * r + 1.f);
  }
  virtual T D2(T r) const override {
    return 528.f * pow(e, 4) * pow(std::max(0.f, 1.f - e * r), 6) +
        (6.f * e * r + 1.f);
  }
  T e;
};

template<typename T> class InverseMultiquadricKernel : public Kernel<T> {
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

template<typename T> class GaussianKernel : public Kernel<T> {
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

template<typename T> class PowKernel : public Kernel<T> {
public:
  PowKernel(T _e) : e(_e) {}
  virtual T operator()(T r) const override { return pow(r, e); }

private:
  T e;
};

} // ponos namespace

#endif // PONOS_BLAS_KERNELS_H
