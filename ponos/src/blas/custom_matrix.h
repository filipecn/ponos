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

#ifndef PONOS_BLAS_CUSTOM_MATRIX_H
#define PONOS_BLAS_CUSTOM_MATRIX_H

namespace ponos {

template <typename T = float> class VectorInterface {
public:
  VectorInterface() {}
  virtual ~VectorInterface() {}
  virtual T operator[](size_t i) const = 0;
  virtual T &operator[](size_t i) = 0;
  virtual size_t size() const = 0;
};

template <typename T = float> class MatrixInterface {
public:
  MatrixInterface() {}
  virtual ~MatrixInterface() {}
  virtual T operator()(size_t i, size_t j) const = 0;
  virtual T &operator()(size_t i, size_t j) = 0;
  virtual void iterateRow(size_t i,
                          std::function<void(const T &, size_t)> f) const = 0;
  virtual void iterateRow(size_t i, std::function<void(T &, size_t)> f) = 0;
  virtual void
  iterateColumn(size_t j, std::function<void(const T &, size_t)> f) const = 0;
  virtual void iterateColumn(size_t j, std::function<void(T &, size_t)> f) = 0;
  virtual size_t rowCount() const = 0;
  virtual size_t columnCount() const = 0;
};

template <typename M, typename T> class MatrixConstAccessor {
public:
  MatrixConstAccessor(const M *_m, bool t = false)
      : m(_m), transposeAccess(t) {}
  virtual ~MatrixConstAccessor() {}
  T operator()(size_t i, size_t j) const { return (*m)(i, j); }
  void iterateRow(size_t i, std::function<void(const T &, size_t)> f) const {
    if (transposeAccess)
      m->iterateColumn(i, f);
    else
      m->iterateRow(i, f);
  }
  size_t rowCount() const { return m->rowCount(); }
  size_t columnCount() const { return m->columnCount(); }

private:
  const M *m;
  bool transposeAccess;
};

/** Performs matrix-vector multiplication
 * \param m **[in]** matrix
 * \param v **[in]** vector
 * \param r **[out]** result of m * v
 */
template <typename M, typename T>
inline void mvm(const MatrixConstAccessor<M, T> &m, const VectorInterface<T> *v,
                VectorInterface<T> *r) {
  /*std::cout << "mvm\n";
  for (size_t i = 0; i < m.rowCount(); i++) {
    for (size_t j = 0; j < m.columnCount(); j++)
      std::cout << m(i, j) << " ";
    std::cout << std::endl;
  }
  std::cout << "mulplyiong by\n";
  for (size_t i = 0; i < v->size(); i++)
    std::cout << (*v)[i] << std::endl;*/
  for (size_t i = 0; i < m.rowCount(); i++) {
    // std::cout << "row " << i << std::endl;
    T s = 0.0;
    m.iterateRow(i, [&s, &v](const T &a, size_t j) {
      // std::cout << "col " << j << " = " << a << "," << (*v)[j] << std::endl;
      s += a * (*v)[j];
    });
    (*r)[i] = s;
  }
  /*std::cout << "answer\n";
  for (size_t i = 0; i < v->size(); i++)
    std::cout << (*r)[i] << std::endl;*/
}

/** Calculates a*x + y.
 * \param a **[in]**
 * \param x **[in]**
 * \param y **[in]**
 * \param r **[out]** result of ax + y
 */
template <typename T>
inline void axpy(T a, const VectorInterface<T> *x, VectorInterface<T> *y,
                 VectorInterface<T> *r) {
  size_t size = x->size();
  for (size_t i = 0; i < size; i++)
    (*r)[i] = a * (*x)[i] + (*y)[i];
}

/** Computes b - Ax
 * \param A **[in]**
 * \param x **[in]**
 * \param b **[in]**
 * \param r **[out]** result of b - Ax
 */
template <typename M, typename T>
inline void residual(const MatrixConstAccessor<M, T> &A,
                     const VectorInterface<T> *x, const VectorInterface<T> *b,
                     VectorInterface<T> *r) {
  // std::cout << "computing residual\n";
  mvm(A, x, r);
  for (size_t i = 0; i < b->size(); i++)
    (*r)[i] = (*b)[i] - (*r)[i];
  /*std::cout << "answer\n";
  for (size_t i = 0; i < r->size(); i++)
    std::cout << (*r)[i] << std::endl;*/
}

template <typename T> inline double norm(const VectorInterface<T> *v) {
  size_t size = v->size();
  T sum = static_cast<T>(0);
  for (size_t i = 0; i < size; i++)
    sum += (*v)[i] * (*v)[i];
  return sqrt(sum);
}

template <typename T> inline T infnorm(const VectorInterface<T> *v) {
  size_t size = v->size();
  T m = static_cast<T>(0);
  for (size_t i = 0; i < size; i++)
    m = std::max(m, static_cast<T>(fabs((*v)[i])));
  return m;
}

template <typename M, typename T>
void dSolve(const MatrixConstAccessor<M, T> &A, const VectorInterface<T> *b,
            VectorInterface<T> *x) {
  for (size_t i = 0; i < A.rowCount(); i++)
    if (A(i, i) != static_cast<T>(0.0))
      (*x)[i] = (*b)[i] / A(i, i);
    else
      (*x)[i] = (*b)[i];
}

template <typename T>
T dot(const VectorInterface<T> *a, const VectorInterface<T> *b) {
  T sum = 0.0;
  for (size_t i = 0; i < a->size(); i++)
    sum += (*a)[i] * (*b)[i];
  return sum;
}
} // ponos namespace

#endif // PONOS_BLAS_CUSTOM_MATRIX_H
