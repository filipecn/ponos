#ifndef PONOS_STRUCTURES_Z_GRID_H
#define PONOS_STRUCTURES_Z_GRID_H

#include "log/debug.h"
#include "structures/grid_interface.h"

#include <algorithm>
#include <functional>
#include <vector>

namespace ponos {

template <class T> class ZGrid : public Grid2DInterface<T> {
public:
  ZGrid();
  ZGrid(uint32_t w, uint32_t h);

  void updateDataStructure() override;

  T getData(int i, int j) const override { return data[morton_code(i, j)]; }
  T &getData(int i, int j) override { return data[morton_code(i, j)]; }

  void reset(std::function<void(T &t)> f) {
    for (uint32_t i = 0; i < data.size(); ++i)
      f(data[i]);
  }

  void setAll(const T t) override {
    for (uint32_t i = 0; i < data.size(); i++)
      data[i] = t;
  }

  T infNorm() const {
    T r = std::fabs((*this)(0, 0));
    ivec2 ij;
    ivec2 D(this->width, this->height);
    FOR_INDICES0_2D(D, ij)
    r = std::max(r, std::fabs((*this)(ij)));
    return r;
  }

  void copyData(const ZGrid<T> *g) {
    std::copy(g->data.begin(), g->data.end(), data.begin());
  }

protected:
  uint32_t separate_by_1(uint32_t n) const {
    n = (n ^ (n << 8)) & 0x00ff00ff;
    n = (n ^ (n << 4)) & 0x0f0f0f0f;
    n = (n ^ (n << 2)) & 0x33333333;
    n = (n ^ (n << 1)) & 0x55555555;
    return n;
  }

  uint32_t morton_code(uint32_t x, uint32_t y) const {
    return (separate_by_1(y) << 1) + separate_by_1(x);
  }

  std::vector<T> data;
};

template <class T> class CZGrid : public ZGrid<T> {
public:
  CZGrid() {}
  CZGrid(uint32_t w, uint32_t h) { this->setDimensions(w, h); }
  T sample(float x, float y) const;
};

template <class T> ZGrid<T>::ZGrid() {}

template <class T> ZGrid<T>::ZGrid(uint32_t w, uint32_t h) {
  this->setDimensions(w, h);
}

template <class T> void ZGrid<T>::updateDataStructure() {
  data.resize(morton_code(this->width, this->height));
}

template <class T> T CZGrid<T>::sample(float x, float y) const {
  Point2 gp = this->toGrid(Point2(x, y));
  int x0 = static_cast<int>(gp.x);
  int y0 = static_cast<int>(gp.y);
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  x0 = std::max(0, std::min(static_cast<int>(this->width) - 1, x0));
  y0 = std::max(0, std::min(static_cast<int>(this->height) - 1, y0));
  x1 = std::max(0, std::min(static_cast<int>(this->width) - 1, x1));
  y1 = std::max(0, std::min(static_cast<int>(this->height) - 1, y1));

  float p[4][4];
  int delta[] = {-1, 0, 1, 2};
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      p[i][j] = this->safeData(x0 + delta[i], y0 + delta[j]);
  return ponos::bicubicInterpolate<float>(p, gp.x - x0, gp.y - y0);
}

} // ponos namespace

#endif
