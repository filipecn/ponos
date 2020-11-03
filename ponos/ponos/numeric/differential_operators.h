/// Copyright (c) 2020, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
/// \file fd_matrix.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date 2020-11-03
///
/// \brief

#ifndef PONOS_PONOS_PONOS_NUMERIC_DIFFERENTIAL_OPERATORS_H
#define PONOS_PONOS_PONOS_NUMERIC_DIFFERENTIAL_OPERATORS_H

#include <ponos/numeric/grid.h>
#include <ponos/numeric/vector_grid.h>

namespace ponos {

enum class DerivativeMode {
  FIRST_ORDER_CENTRAL_FD, //!< (f(x + h) - f(x - h)) / 2h
  FIRST_ORDER_FORWARD_FD, //!< (f(x + h) - f(x)) / h
  FIRST_ORDER_BACKWARD_FD, //!< (f(x) - f(x - h)) / h
  SECOND_ORDER_CENTRAL_FD, //!< (f(x + h) - 2f(x) + f(x - h)) / h2
  SECOND_ORDER_FORWARD_FD, //!< (f(x + 2h) - 2f(x + h) + f(x)) / h2
  SECOND_ORDER_BACKWARD_FD, //!< (f(x) - 2f(x - h) + f(x - 2h)) / h2
};

/// Set of differential operators
class DiffOps {
public:
  ///
  /// \tparam T
  /// \param field
  /// \param grad_field
  /// \param derivative_mode
  /// \param boundary_derivative_mode
  template<typename T>
  static void grad(ConstGrid2Accessor<T> field, VectorGrid2<T> &grad_field,
                   DerivativeMode derivative_mode = DerivativeMode::SECOND_ORDER_CENTRAL_FD,
                   DerivativeMode boundary_derivative_mode = DerivativeMode::SECOND_ORDER_BACKWARD_FD) {
    const real_t hy_rec = static_cast<real_t>(1) / (field.spacing().y);
    const real_t hx_rec = static_cast<real_t>(1) / (field.spacing().x);
    const real_t hy2_rec = static_cast<real_t>(1) / (field.spacing().y * field.spacing().y);
    const real_t hx2_rec = static_cast<real_t>(1) / (field.spacing().x * field.spacing().x);
    const real_t two_hy_rec = static_cast<real_t>(1) / (2 * field.spacing().y);
    const real_t two_hx_rec = static_cast<real_t>(1) / (2 * field.spacing().x);

    using E = const typename ConstGrid2Iterator<T>::Element &;

    auto getDs = [&](std::function<T(E)> &DX, std::function<T(E)> &DY, DerivativeMode mode) {
      switch (mode) {
      case DerivativeMode::FIRST_ORDER_CENTRAL_FD:
        DX = [&two_hx_rec](E e) -> T {
          return (e.right() - e.left()) * two_hx_rec;
        };
        DY = [&two_hy_rec](E e) -> T { return (e.top() - e.bottom()) * two_hy_rec; };
        break;
      case DerivativeMode::FIRST_ORDER_FORWARD_FD:DX = [&hx_rec](E e) -> T { return (e.right() - e.value) * hx_rec; };
        DY = [&hy_rec](E e) -> T { return (e.top() - e.value) * hy_rec; };
        break;
      case DerivativeMode::FIRST_ORDER_BACKWARD_FD:DX = [&hx_rec](E e) -> T { return (e.value - e.left()) * hx_rec; };
        DY = [&hy_rec](E e) -> T { return (e.value - e.bottom()) * hy_rec; };
        break;
      case DerivativeMode::SECOND_ORDER_CENTRAL_FD:
        DX = [&hx2_rec](E e) -> T {
          return (e.right() - 2 * e.value + e.left()) * hx2_rec;
        };
        DY = [&hy2_rec](E e) -> T { return (e.top() - 2 * e.value + e.bottom()) * hy2_rec; };
        break;
      case DerivativeMode::SECOND_ORDER_FORWARD_FD:
        DX = [&hx2_rec](E e) -> T {
          return (e.right(2) - 2 * e.right() + e.value) * hx2_rec;
        };
        DY = [&hy2_rec](E e) -> T {
          return (e.top(2) - 2 * e.top() + e.value) * hy2_rec;
        };
        break;
      case DerivativeMode::SECOND_ORDER_BACKWARD_FD:
        DX = [&hx2_rec](E e) -> T {
          return (e.value - 2 * e.left() + e.left(2)) * hx2_rec;
        };
        DY = [&hy2_rec](E e) -> T { return (e.value - 2 * e.bottom() + e.bottom(2)) * hy2_rec; };
      };
    };

    std::function<T(E)> Dx, Dy, lowerBx, lowerBy, upperBx, upperBy;
    getDs(Dx, Dy, derivative_mode);
    if (boundary_derivative_mode == DerivativeMode::FIRST_ORDER_CENTRAL_FD ||
        boundary_derivative_mode == DerivativeMode::SECOND_ORDER_CENTRAL_FD) {
      getDs(lowerBx, lowerBy, boundary_derivative_mode);
      getDs(upperBx, upperBy, boundary_derivative_mode);
    } else if (boundary_derivative_mode == DerivativeMode::FIRST_ORDER_FORWARD_FD ||
        boundary_derivative_mode == DerivativeMode::FIRST_ORDER_BACKWARD_FD) {
      getDs(lowerBx, lowerBy, DerivativeMode::FIRST_ORDER_FORWARD_FD);
      getDs(upperBx, upperBy, DerivativeMode::FIRST_ORDER_BACKWARD_FD);
    } else {
      getDs(lowerBx, lowerBy, DerivativeMode::SECOND_ORDER_FORWARD_FD);
      getDs(upperBx, upperBy, DerivativeMode::SECOND_ORDER_BACKWARD_FD);
    }

    auto u = grad_field.u().accessor();
    for (auto f :  field) {
      if (f.index.i == 0)
        u[f.index] = lowerBx(f);
      else if (f.index.i == field.resolution().width - 1)
        u[f.index] = upperBx(f);
      else
        u[f.index] = Dx(f);
    }
    auto v = grad_field.v().accessor();
    for (auto f :  field) {
      if (f.index.j == 0)
        v[f.index] = lowerBy(f);
      else if (f.index.j == field.resolution().height - 1)
        v[f.index] = upperBy(f);
      else
        v[f.index] = Dy(f);
    }
  }
  ///
  /// \tparam T
  /// \param field
  /// \param derivative_mode
  /// \param boundary_derivative_mode
  /// \return
  template<typename T>
  static VectorGrid2<T> grad(ConstGrid2Accessor<T> field,
                             DerivativeMode derivative_mode = DerivativeMode::SECOND_ORDER_CENTRAL_FD,
                             DerivativeMode boundary_derivative_mode = DerivativeMode::SECOND_ORDER_BACKWARD_FD) {
    VectorGrid2<T> grad_field(field.resolution(), field.spacing(), field.origin());
    grad<T>(field, grad_field, derivative_mode, boundary_derivative_mode);
    return grad_field;
  }
};

}

#endif //PONOS_PONOS_PONOS_NUMERIC_DIFFERENTIAL_OPERATORS_H
