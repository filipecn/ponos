#pragma once

#include "log/debug.h"
#include "structures/c_grid_interface.h"

#include <functional>
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
    float sample(float x, float y, std::function<float(T& t)> f) const;
    T dSample(float x, float y, T r);

    void reset(std::function<void(T& t)> f) {
      for (uint32_t i = 0; i < data.size(); ++i)
        f(data[i]);
    }

    void setAll(const T t) {
      for(uint32_t i = 0; i < data.size(); i++)
        data[i] = t;
    }

  private:
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

  template<class T>
  ZGrid<T>::ZGrid() {}

  template<class T>
  ZGrid<T>::ZGrid(uint32_t w, uint32_t h) {
    this->setDimensions(w,h);
    init();
  }

  template<class T>
  void ZGrid<T>::init() {
    data.resize(morton_code(this->width, this->height));
  }

  template<class T>
  T& ZGrid<T>::operator()(uint32_t i, uint32_t j) {
    uint32_t ind = morton_code(i, j);
    if(ind < 0 || ind >= data.size()) {
      return this->border;
    }
    return data[ind];
  }

  template<class T>
  T ZGrid<T>::operator()(uint32_t i, uint32_t j) const {
    uint32_t ind = morton_code(i, j);
    if(ind < 0 || ind >= data.size()) {
      return this->border;
    }
    return data[ind];
  }

  template<class T>
  float ZGrid<T>::sample(float x, float y, std::function<float(T& t)> f) const {
    Point2 gp = this->toGrid(Point2(x, y));
    int x0 = static_cast<int>(gp.x);
    int y0 = static_cast<int>(gp.y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = std::max(0, std::min(static_cast<int>(this->width) - 1, x0));
    y0 = std::max(0, std::min(static_cast<int>(this->height) - 1, y0));
    x1 = std::max(0, std::min(static_cast<int>(this->width) - 1, x1));
    y1 = std::max(0, std::min(static_cast<int>(this->height) - 1, y1));

    Point2 wp = this->toWorld(Point2(x0, y0));
    Vector2 scale = this->toWorld.getScale();

    float p[4][4];
    //int delta[] = {-1, 0, 1, 2};
    //for(int i = 0; i < 4; i++)
    //for(int j = 0; j < 4; j++)
    // p[i][j] = f(safeData(x0 + delta[i], y0 + delta[j]));
    return ponos::bicubicInterpolate<float>(p, (x - wp.x) / scale.x, (y - wp.y) / scale.y);
  }

  template<typename T>
  T ZGrid<T>::dSample(float x, float y, T r){
    ponos::Point<int, 2> gp = this->cell(Point2(x,y));
    if (!this->belongs(gp))
    return r;
    int ind = morton_code(gp[0], gp[1]);
    return data[ind];
  }

  template<typename T>
  T ZGrid<T>::safeData(uint32_t i, uint32_t j) const{
    uint32_t ind = morton_code(std::max(static_cast<uint32_t>(0), std::min(this->width - 1, i)),
                               std::max(static_cast<uint32_t>(0), std::min(this->height - 1, j)));
    return data[ind];
  }

} // ponos namespace
