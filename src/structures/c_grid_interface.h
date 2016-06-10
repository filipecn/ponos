#pragma once

#include "geometry/point.h"
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

    ponos::Point<int, 2> cell(ponos::Point2 wp) {
      ponos::vec2 delta(0.5f, 0.5f);
      return ponos::Point<int, 2>(toGrid(wp) + delta);
    }

    ponos::Point<int, 2> safeCell(ponos::Point2 wp) {
      ponos::vec2 delta(0.5f, 0.5f);
      ponos::Point<int, 2> gp(toGrid(wp) + delta);
      gp[0] = std::min(width - 1, std::max(0, gp[0]));
      gp[1] = std::min(height - 1, std::max(0, gp[1]));
      return gp;
    }

    bool belongs(ponos::Point<int, 2> c) {
      return 0 <= c[0] && c[0]< width && 0 <= c[1] && c[1] < height;
    }

    T border;
    bool useBorder;

    uint32_t width, height;
    Vector2 offset;
    Transform2D toWorld, toGrid;
  };

} // ponos namespace
