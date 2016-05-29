#pragma once

#include "geometry/transform.h"
using ponos::Transform2D;
#include "geometry/vector.h"
using ponos::Vector2;

namespace ponos {

  template<class T>
  class CGrid2DInterface {
  public:
    void setTransform(Vector2 _offset, Vector2 _cellSize) {
      offset = _offset;
      toWorld.reset();
      toWorld.scale(_cellSize.x, _cellSize.y);
      toWorld.translate(offset);
      toWorld.computeInverse();
      toGrid = inverse(toWorld);
    }

    void setDimensions(uint32_t w, uint32_t h) {
      width = w;
      height = h;
    }

    void set(uint32_t w, uint32_t h, Vector2 _offset, Vector2 _cellSize) {
      setTransform(_offset, _cellSize);
      setDimensions(w, h);
    }

    T border;
    bool useBorder;

    uint32_t width, height;
    Vector2 offset;
    Transform2D toWorld, toGrid;
  };

} // ponos namespace
