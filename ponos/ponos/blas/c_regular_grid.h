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

#ifndef PONOS_BLAS_C_REGULAR_GRID_H
#define PONOS_BLAS_C_REGULAR_GRID_H

#include <ponos/blas/field.h>
#include <ponos/common/macros.h>
#include <ponos/log/debug.h>
#include <ponos/structures/grid_interface.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace ponos {

/** Simple matrix structure.
 */
template <typename T = float> class CRegularGrid : public CGridInterface<T> {
public:
  CRegularGrid() {}
  /* Constructor
   * \param d **[in]** dimensions
   * \param b **[in]** background (default value)
   * \param cellSize **[in | optional]** grid spacing
   * \param offset **[in | optional]** grid origin
   */
  CRegularGrid(const ivec3 &d, const T &b, const vec3 cellSize = vec3(1.f),
               const vec3 &offset = vec3());
  CRegularGrid(const ivec3 &d, const T &b, const BBox &bb);

  ~CRegularGrid();
  /* @inherit */
  void set(const ivec3 &i, const T &v) override;
  void setAll(T v);
  /* @inherit */
  T operator()(const ivec3 &i) const override {
    CHECK_IN_BETWEEN(i[0], 0, this->dimensions[0]);
    CHECK_IN_BETWEEN(i[1], 0, this->dimensions[1]);
    CHECK_IN_BETWEEN(i[2], 0, this->dimensions[2]);
    return data[i[0]][i[1]][i[2]];
  }
  /* @inherit */
  T &operator()(const ivec3 &i) override {
    CHECK_IN_BETWEEN(i[0], 0, this->dimensions[0]);
    CHECK_IN_BETWEEN(i[1], 0, this->dimensions[1]);
    CHECK_IN_BETWEEN(i[2], 0, this->dimensions[2]);
    return data[i[0]][i[1]][i[2]];
  }
  /* @inherit */
  T operator()(const uint &i, const uint &j, const uint &k) const override {
    CHECK_IN_BETWEEN(static_cast<int>(i), 0, this->dimensions[0]);
    CHECK_IN_BETWEEN(static_cast<int>(j), 0, this->dimensions[1]);
    CHECK_IN_BETWEEN(static_cast<int>(k), 0, this->dimensions[2]);
    return data[i][j][k];
  }
  /* @inherit */
  T &operator()(const uint &i, const uint &j, const uint &k) override {
    CHECK_IN_BETWEEN(static_cast<int>(i), 0, this->dimensions[0]);
    CHECK_IN_BETWEEN(static_cast<int>(j), 0, this->dimensions[1]);
    CHECK_IN_BETWEEN(static_cast<int>(k), 0, this->dimensions[2]);
    return data[i][j][k];
  }
  T safeData(int i, int j, int k) const;
  T operator()(const float &x, const float &y, const float &z) const override {
    Point3 gp = this->toGrid(ponos::Point3(x, y, z));
    float p[3] = {gp.x, gp.y, gp.z};
    return trilinearInterpolate<T>(p, data, this->background,
                                   this->dimensions.v);
  }
  T operator()(const vec3 &i) const override {
    return (*this)(i[0], i[1], i[2]);
  }
  void normalize() override;
  void normalizeElements() override;

private:
  T ***data;
};
/**
 */
template <typename T = float>
class CRegularGrid2D : public Grid2DInterface<T>,
                       virtual public FieldInterface2D<T> {
public:
  CRegularGrid2D();
  CRegularGrid2D(uint32_t w, uint32_t h);
  virtual ~CRegularGrid2D() {}

  T getData(int i, int j) const override { return data[i][j]; }
  T &getData(int i, int j) override { return data[i][j]; }
  void updateDataStructure() override;
  T sample(float x, float y) const override;

  T infNorm() const {
    T r = std::fabs((*this)(0, 0));
    for (uint32_t i = 0; i < data.size(); i++)
      for (uint32_t j = 0; j < data[i].size(); ++j)
        r = std::max(r, std::fabs(data[i][j]));
    return r;
  }
  void copyData(const CRegularGrid2D<T> *g) {
    for (size_t i = 0; i < data.size(); i++)
      std::copy(g->data[i].begin(), g->data[i].end(), data[i].begin());
  }

protected:
  std::vector<std::vector<T>> data;
};

#include "c_regular_grid.inl"

} // ponos namespace

#endif // PONOS_BLAS_C_REGULAR_GRID_H
