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

#ifndef PONOS_COMMON_ITERATORS_H
#define PONOS_COMMON_ITERATORS_H

#include <ponos/geometry/vector.h>

namespace ponos {

template <int D> class IndexIterator {
public:
  IndexIterator(Vector<int, D> start, Vector<int, D> end,
                Vector<int, D> step = Vector<int, D>(1))
      : start_(start), curr_(start), end_(end), step_(step) {}
  IndexIterator(Vector<int, D> end) : end_(end) {
    for (int d = 0; d < D; d++)
      step_[d] = (end_[d] < 0) ? -1 : 1;
  }

  size_t count() const {
    Vector<int, D> dif(0);
    for (int d = 0; d < D; d++)
      dif[d] = std::abs(end_[d] - start_[d]) + 1;
    if (dif == Vector<int, D>(0))
      return 0;
    size_t c = 1;
    for (int d = 0; d < D; d++)
      if (dif[d])
        c *= dif[d];
    return c;
  }

  size_t id() const { return id_; }

  bool next() const {
    return curr_ != end_;
    for (int d = D - 1; d >= 0; d--)
      if ((step_[d] < 0 && curr_[d] >= end_[d] + 1) ||
          (step_[d] > 0 && curr_[d] <= end_[d] - 1))
        return true;
    return false;
  }
  Vector<int, D> operator()() { return curr_; }
  IndexIterator<D> &operator++() {
    bool carry = true;
    for (int d = 0; d < D && carry; d++) {
      if (carry)
        curr_[d] += step_[d];
      carry = false;
      if ((step_[d] < 0 && curr_[d] < end_[d]) ||
          (step_[d] > 0 && curr_[d] > end_[d])) {
        curr_[d] = start_[d];
        carry = true;
      }
    }
    id_++;
    return *this;
  }
  IndexIterator<D> operator++(int) {
    IndexIterator<D> cpy(*this);
    ++(*this);
    return cpy;
  }
  void operator+=(Vector<int, D> step) { curr_ += step; }

private:
  Vector<int, D> start_ = Vector<int, D>(0), curr_, end_,
                 step_ = Vector<int, D>(1);
  size_t id_ = 0;
};

} // namespace ponos

#endif // PONOS_COMMON_ITERATORS_H