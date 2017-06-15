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

#ifndef PONOS_BLAS_SPARSE_VECTOR_H
#define PONOS_BLAS_SPARSE_VECTOR_H

#include <algorithm>
#include <functional>
#include <vector>

namespace ponos {

template <typename T> class SparseVector {
public:
  SparseVector(size_t _n, size_t mcount = 0) : n(_n), count(0) {
    if (mcount)
      data.resize(mcount);
  }
  void insert(size_t i, const T &v) {
    if (count >= data.size())
      data.emplace_back();
    data[count].value = v;
    data[count].rowIndex = i;
    count++;
  }
  void iterate(std::function<void(T &v, size_t i)> f) {
    for (size_t r = 0; r < count; r++)
      f(data[r].value, data[r].rowIndex);
  }
  size_t elementCount() const { return count; }
  size_t size() const { return n; }
  void copy(const SparseVector<T> &v) {
    data.resize(v.data.size());
    std::copy(v.data.begin(), v.data.end(), data.begin());
    count = v.count;
    n = v.n;
  }
  T dot(const SparseVector<T> &B) const {
    size_t a = 0, b = 0;
    T r = 0.0;
    while (a < count && b < B.count) {
      if (data[a].rowIndex == B.data[b].rowIndex)
        r += data[a++].value * B.data[b++].value;
      else if (data[a].rowIndex < B.data[b].rowIndex)
        a++;
      else
        b++;
    }
    return r;
  }
  template <typename S>
  friend void axpy(S a, const SparseVector<S> &x, const SparseVector<S> &y,
                   SparseVector<S> *r);
  template <typename S>
  friend void axpy(S a, SparseVector<S> *x, const SparseVector<S> &y);

  class const_iterator {
  public:
    const_iterator(const SparseVector<T> &_v) : v(_v), cur(0) {}
    bool next() const { return cur < v.count; }
    T value() { return v.data[cur].value; }
    size_t rowIndex() const { return v.data[cur].rowIndex; }
    void operator++() { cur++; }
    int count() const { return v.count; }

  private:
    const SparseVector<T> &v;
    size_t cur;
  };

  class iterator {
  public:
    iterator(SparseVector<T> &_v) : v(_v), cur(0) {}
    bool next() const { return cur < v.count; }
    T *value() { return v.data[cur].value; }
    size_t rowIndex() const { return v.data[cur].rowIndex; }
    void operator++() { cur++; }
    int count() const { return v.count; }

  private:
    SparseVector<T> &v;
    size_t cur;
  };

  // private:
  struct Element {
    T value;
    size_t rowIndex;
  };
  T dummy;
  size_t n, count;
  std::vector<Element> data;
};

typedef SparseVector<double> svecd;

template <typename T>
inline void axpy(T a, const SparseVector<T> &x, const SparseVector<T> &y,
                 SparseVector<T> *r) {
  r->count = 0;
  size_t i = 0, j = 0;
  while (i < x.count || j < y.count) {
    if (i >= x.count)
      r->insert(y.data[j].rowIndex, y.data[j].value), j++;
    else if (j >= y.count)
      r->insert(x.data[i].rowIndex, x.data[i].value * a), i++;
    else if (x.data[i].rowIndex < y.data[j].rowIndex)
      r->insert(x.data[i].rowIndex, x.data[i].value * a), i++;
    else if (y.data[j].rowIndex < x.data[i].rowIndex)
      r->insert(y.data[j].rowIndex, y.data[j].value), j++;
    else
      r->insert(y.data[j].rowIndex, a * x.data[i].value + y.data[j].value), i++,
          j++;
  }
}
template <typename T>
inline void axpy(T a, SparseVector<T> *x, const SparseVector<T> &y) {
  size_t i = 0, j = 0;
  size_t xcount = x->count;
  while (i < xcount || j < y.count) {
    if (i >= xcount)
      x->insert(y.data[j].rowIndex, y.data[j].value), j++;
    else if (j >= y.count)
      x->data[i++].value *= a;
    else if (x->data[i].rowIndex < y.data[j].rowIndex)
      x->data[i++].value *= a;
    else if (y.data[j].rowIndex < x->data[i].rowIndex)
      x->insert(y.data[j].rowIndex, y.data[j].value), j++;
    else
      x->data[i].value = a * x->data[i].value + y.data[j++].value, i++;
  }
  if (xcount < x->count) {
    std::sort(x->data.begin(), x->data.end(),
              [](const typename SparseVector<T>::Element &a,
                 const typename SparseVector<T>::Element &b)
                  -> bool { return a.rowIndex < b.rowIndex; });
  }
}

} // ponos namespace

#endif // PONOS_BLAS_SPARSE_VECTOR_H
