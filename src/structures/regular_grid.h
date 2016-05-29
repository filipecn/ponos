#pragma once

#include "log/debug.h"
#include "structures/c_grid_interface.h"

#include <vector>

namespace ponos {

  template<class T>
  class RegularGrid : public CGrid2DInterface<T> {
  public:
    RegularGrid();
    RegularGrid(uint32_t w, uint32_t h);

    virtual ~RegularGrid();

    T& operator() (int i, int j) override;
    T operator() (int i, int j) const override;
    T safeData(int i, int j) const override;

    void set(uint32_t w, uint32_t h, Vector2 offset, Vector2 cellSize);
    void setAll(T v);

  private:
    std::vector<std::vector<T> >data;
  };

  template<typename T>
  RegularGrid<T>::RegularGrid(){
    width = height = 0;
    useBorder = false;
  }

  template<typename T>
  RegularGrid<T>::RegularGrid(uint32_t w, uint32_t h) {
    width = w;
    height = h;
  }

  template<typename T>
  RegularGrid<T>::~RegularGrid() {}

  template<typename T>
  void RegularGrid<T>::set(uint32_t w, uint32_t h, Vector2 offset, Vector2 cellSize) {
    set(w, h);
    set(offset, cellSize);
    data.resize(w, std::vector<T>());
    for (int i = 0; i < w; i++)
     data[i].resize(h);
  }

  template<typename T>
  T& RegularGrid<T>::operator() (int i, int j) {
    CHECK_IN_BETWEEN(i, 0, width);
    CHECK_IN_BETWEEN(j, 0, height);
    return data[i][j];
  }

  template<typename T>
  T RegularGrid<T>::operator() (int i, int j) const {
    CHECK_IN_BETWEEN(i, 0, width);
    CHECK_IN_BETWEEN(j, 0, height);
    return data[i][j];
  }

  template<typename T>
  void RegularGrid<T>::setAll(T v){
    for (int i = 0; i < width; i++)
    for (int j = 0; j < height; j++)
    data[i][j] = v;
  }

  template<typename T>
  T RegularGrid<T>::safeData(int i, int j) const{
    return data[max(0, min(width-1,i))][max(0, min(height-1,j))];
  }
}  // ponos namespace
