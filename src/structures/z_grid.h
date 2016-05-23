#pragma once

#include "log/debug.h"
#include "structures/c_grid_interface.h"

#include <vector>

namespace ponos {

  template<class T>
  class ZGrid : public CGrid2DInterface<T> {
  public:
    ZGrid();
    ZGrid(uint32_t w, uint32_t h);

    void init();

    T& operator()(uint32_t i, uint32_t j);
    T operator() (uint32_t i, uint32_t j) const;
    T safeData(uint32_t i, uint32_t j) const;

  private:
    uint32_t separate_by_1(uint32_t n) {
      n = (n ^ (n << 8)) & 0x00ff00ff;
      n = (n ^ (n << 4)) & 0x0f0f0f0f;
      n = (n ^ (n << 2)) & 0x33333333;
      n = (n ^ (n << 1)) & 0x55555555;
      return n;
    }

    uint32_t morton_code(uint32_t x, uint32_t y) {
      return (separate_by_1(y) << 1) + separate_by_1(x);
    }

    std::vector<T> data;
  };

  template<class T>
  ZGrid<T>::ZGrid() {}

  template<class T>
  ZGrid<T>::ZGrid(uint32_t w, uint32_t h) {
    setDimensions(w,h);
    init();
  }

  template<class T>
  void ZGrid<T>::init() {
    data.resize(width*height);
  }

  template<class T>
  T& ZGrid<T>::operator()(uint32_t i, uint32_t j) {
    uint32_t ind = morton_code(i, j);
    return data[ind];
  }

}; // ponos namespace
