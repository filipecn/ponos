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

#ifndef PONOS_INTERPOLATOR_INTERFACE_H
#define PONOS_INTERPOLATOR_INTERFACE_H

#include <ponos/blas/kernels.h>
#include <vector>

namespace ponos {
/// Interpolator interface
/// \tparam P position type (Point3, Point2, ...)
/// \tparam T data type
template<class P, typename T>
class InterpolatorInterface {
public:
  InterpolatorInterface() = default;
  virtual ~InterpolatorInterface() = default;
  /// \param target sample position
  /// \param points list of points
  /// \param values list of values stored in each point
  /// \return interpolated value at target position
  virtual T interpolateAt(P target, const std::vector<P> & points,
                               const std::vector<T> & values) = 0;
  /// \param target sample position
  /// \param points list of points
  /// \return weights assigned to each point on interpolation process
  virtual std::vector<double> weights(P target, const std::vector<P> &points) const = 0;
};

/// Performs an RBF interpolation
/// \tparam P position type (Point3, Point2, ...)
/// \tparam T data type
template<class P, typename T>
class RBFInterpolator : public InterpolatorInterface<P, T> {
public:
  /// \param kernel **[optional]** radial bassis function
  explicit RBFInterpolator(Kernel<T>* kernel = new GaussianKernel<T>(1.));
  ~RBFInterpolator();
  T interpolateAt(P target, const std::vector<P> & points,
                       const std::vector<T> & values) override;
  std::vector<double> weights(P target, const std::vector<P> &points) const override;

private:
  Kernel<T> *kernel_; ///< radial basis kernel
};

#include "interpolator.inl"

} // ponos namespace

#endif //PONOS_INTERPOLATOR_INTERFACE_H
