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

#ifndef PONOS_STRUCTURES_REGULAR_GRID_H
#define PONOS_STRUCTURES_REGULAR_GRID_H

#include <ponos/common/macros.h>
#include <ponos/geometry/vector.h>
#include <ponos/log/debug.h>
#include <ponos/structures/grid_interface.h>

#include <memory>
#include <vector>

namespace ponos {

/* Regular grid
 *
 * Simple matrix structure.
 */
template <class T = int> class RegularGrid {
public:
  RegularGrid() {}
  /* Constructor
   * @d **[in]** dimensions
   * @b **[in]** background (default value)
   */
  RegularGrid(const ivec3 &d, const T &b) { set(d, b); }
  ~RegularGrid() { clear(); }
  void set(const ivec3 &d, const T &b) {
    clear();
    dimensions = d;
    background = b;
    bdummy = b;
    data = new T **[d[0]];
    int i;
    FOR_LOOP(i, 0, d[0])
    data[i] = new T *[d[1]];
    ponos::ivec2 dd = d.xy();
    ponos::ivec2 ij;
    FOR_INDICES0_2D(dd, ij)
    data[ij[0]][ij[1]] = new T[d[2]];
  }
  T operator()(const ivec3 &i) const {
    if (i >= ivec3() && i < dimensions)
      return data[i[0]][i[1]][i[2]];
    return background;
  }
  T &operator()(const ivec3 &i) {
    if (i >= ivec3() && i < dimensions)
      return data[i[0]][i[1]][i[2]];
    return bdummy;
  }
  T operator()(const int &i, const int &j, const int &k) const {
    if (i >= 0 && i < dimensions[0] && j >= 0 && i < dimensions[1] && k >= 0 &&
        i < dimensions[2])
      return data[i][j][k];
    return background;
  }
  T &operator()(const int &i, const int &j, const int &k) {
    if (i >= 0 && i < dimensions[0] && j >= 0 && i < dimensions[1] && k >= 0 &&
        i < dimensions[2])
      return data[i][j][k];
    return bdummy;
  }

private:
  void clear() {
    ponos::ivec2 d = dimensions.xy();
    ponos::ivec2 ij;
    FOR_INDICES0_2D(d, ij)
    delete[] data[ij[0]][ij[1]];
    int i;
    FOR_LOOP(i, 0, dimensions[0])
    delete[] data[i];
    delete data;
  }

  ponos::ivec3 dimensions;
  T ***data;
  T background, bdummy;
};

/** Regular 2D grid
 */
template <class T> class RegularGrid2D : public Grid2DInterface<T> {
public:
  RegularGrid2D() : data(nullptr) {}
  /* Constructor
   * @d **[in]** dimensions
   * @b **[in]** border (default value)
   */
  RegularGrid2D(const ivec2 &d, const T &b) : data(nullptr) {
    UNUSED_VARIABLE(b);
    this->set(d);
  }
  RegularGrid2D(size_t w, size_t h, T v) {
    this->setDimensions(w, h);
    this->setAll(v);
  }
  ~RegularGrid2D() { clear(); }

  RegularGrid2D(uint32_t w, uint32_t h) : data(nullptr) {
    this->setDimensions(w, h);
  }

  T getData(int i, int j) const override { return data[i][j]; }
  T &getData(int i, int j) override { return data[i][j]; }

  void updateDataStructure() override {
    // TODO fix that!
    // clear();
    data = new T *[this->width];
    uint32_t i;
    FOR_LOOP(i, 0, this->width)
    data[i] = new T[this->height];
  }

private:
  void clear() {
    if (data == nullptr)
      return;
    uint32_t i;
    FOR_LOOP(i, 0, this->width)
    delete[] data[i];
    delete[] data;
    data = nullptr;
  }

  T **data;
};

} // ponos namespace

#endif
