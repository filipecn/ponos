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

#ifndef PONOS_BLAS_SPARSE_MATRIX_H
#define PONOS_BLAS_SPARSE_MATRIX_H

#include <vector>

namespace ponos {
/** Implements a compressed row storage
 */
template <typename T = double> class SparseMatrix {
public:
  SparseMatrix() : n(0), m(0), count(0) {}
  SparseMatrix(size_t _n, size_t _m, size_t mcount) : n(_n), m(_m), count(0) {
    data.resize(mcount);
    rowPtr.resize(n + 1, -1);
  }
  void set(size_t _n, size_t _m, size_t mcount) {
    n = _n;
    m = _m;
    count = 0;
    data.resize(mcount);
    rowPtr.resize(n + 1, -1);
  }
  void insert(size_t i, size_t j, const T &v) {
    static int lastRow = -1;
    if (count >= data.size())
      return;
    if (lastRow != static_cast<int>(i)) {
      rowPtr[i] = count;
      lastRow = i;
    }
    data[count].value = v;
    data[count].columnIndex = j;
    count++;
    rowPtr[n] = count;
  }
  int valuesInRowCount(size_t i) const {
    if (rowPtr[i] < 0)
      return 0;
    size_t k = i + 1;
    while (k < n && rowPtr[k] < 0)
      k++;
    return rowPtr[k] - rowPtr[i];
  }
  void iterateRow(size_t r, std::function<void(T &v, size_t j)> f) {
    int rc = valuesInRowCount(r);
    for (int k = rowPtr[r]; k < rowPtr[r] + rc; k++)
      f(data[k].value, data[k].columnIndex);
  }
  void iterateRow(size_t r, std::function<void(const T &v, size_t j)> f) const {
    int rc = valuesInRowCount(r);
    for (int k = rowPtr[r]; k < rowPtr[r] + rc; k++)
      f(data[k].value, data[k].columnIndex);
  }
  void printDense() const {
    for (size_t r = 0; r < n; r++) {
      size_t column = 0;
      iterateRow(r, [&column](const T &v, size_t c) {
        if (c > column)
          while (column < c) {
            std::cout << " 0.0000 ";
            column++;
          }
        if (v >= 0.0)
          std::cout << " " << v << " ";
        else
          std::cout << v << " ";
        column++;
      });
      while (column++ < m)
        std::cout << " 0.0000 ";
      std::cout << std::endl;
    }
  }
  size_t rowCount() const { return n; }
  size_t columnCount() const { return m; }
  void copy(const SparseMatrix<T> &mat) {
    rowPtr.resize(mat.rowPtr.size());
    data.resize(mat.data.size());
    n = mat.n;
    m = mat.m;
    count = mat.count;
    std::copy(mat.rowPtr.begin(), mat.rowPtr.end(), rowPtr.begin());
    std::copy(mat.data.begin(), mat.data.end(), data.begin());
  }
  class const_iterator {
  public:
    const_iterator(const SparseMatrix<T> &_m) : m(_m), cur(0), curRow(0) {
      updateCurRow();
    }
    bool next() const { return cur < m.data.size(); }
    T value() { return m.data[cur].value; }
    size_t rowIndex() const { return curRow; }
    size_t columnIndex() const { return m.data[cur].columnIndex; }
    void operator++() {
      cur++;
      updateCurRow();
    }

  private:
    void updateCurRow() {
      size_t i = curRow, last = curRow;
      while (i < m.rowPtr.size() && m.rowPtr[i] <= static_cast<int>(cur)) {
        i++;
        if (m.rowPtr[i] >= 0 && m.rowPtr[i] <= static_cast<int>(cur))
          last = i;
      }
      curRow = last;
    }
    const SparseMatrix<T> &m;
    size_t cur;
    size_t curRow;
  };

private:
  struct Element {
    T value;
    size_t columnIndex;
  };
  T dummy;
  size_t n, m, count;
  int rowIndex(size_t i, size_t j) {
    if (rowPtr[i] < 0)
      return -1;
    for (int k = rowPtr[i]; k < rowPtr[i] + 1; k++)
      if (data[k].rowIndex == i)
        return k;
    return -1;
  }
  std::vector<int> rowPtr;
  std::vector<Element> data;
};

typedef SparseMatrix<double> smatrixd;
typedef SparseMatrix<float> smatrixf;

} // ponos namespace

#endif // PONOS_BLAS_SPARSE_MATRIX_H
