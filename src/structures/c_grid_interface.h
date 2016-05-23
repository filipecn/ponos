#pragma once

#include "geometry/transform.h"
using ponos::Transform2D;
#include "geometry/vector.h"
using ponos::Vector2;

namespace ponos {

  template<class T>
  class CGrid2DInterface {
  public:
    void setTransform(Vector2 offset, Vector2 cellSize) {
      toWorld.reset();
      toWorld.scale(cellSize.x, cellSize.y);
      toWorld.translate(offset);
    }

    void setDimensions(uint32_t w, uint32_t h) {
      width = w;
      height = h;
    }

    T border;
    bool useBorder;

  protected:
    uint32_t width, height;
    Vector2 offset;
    Transform2D toWorld;
  };

} // ponos namespace
