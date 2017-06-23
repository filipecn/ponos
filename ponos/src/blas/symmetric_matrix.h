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

#ifndef PONOS_BLAS_SYMMETRIC_MATRIX_H
#define PONOS_BLAS_SYMMETRIC_MATRIX_H

#include <vector>

namespace ponos {

template <typename T = double> class SymmetricMatrix {
public:
  SymmetricMatrix() : n(0) {}
  SymmetricMatrix(size_t _n) { set(_n); }
  void set(size_t _n) {
    n = _n;
    data.resize(static_cast<size_t>(n * (1 + n) * .5));
    size_t sum = 0;
    for (size_t i = n; i > 0; i--) {
      rowIndex.emplace_back(sum);
      sum += i;
    }
  }
  T &operator()(size_t i, size_t j) { return data[vIndex(i, j)]; }
  T operator()(size_t i, size_t j) const { return data[vIndex(i, j)]; }

private:
  size_t vIndex(size_t i, size_t j) const {
    // 00 01 02 03
    // 10 11 12 13
    // 20 21 22 23
    // 30 31 32 33
    if (i > j)
      return vIndex(j, i);
    return rowIndex[i] + j - i;
  }
  size_t n;
  std::vector<T> data;
  std::vector<size_t> rowIndex;
};

} // ponos namespace

#endif // PONOS_BLAS_SYMMETRIC_MATRIX_H
