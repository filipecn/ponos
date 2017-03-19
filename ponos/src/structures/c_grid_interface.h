#ifndef PONOS_STRUCTURES_C_GRID_INTERFACE_H
#define PONOS_STRUCTURES_C_GRID_INTERFACE_H

#include "geometry/bbox.h"
#include "geometry/point.h"
#include "geometry/transform.h"
#include "geometry/vector.h"

#include <algorithm>

namespace ponos {

enum class CGridAccessMode { CLAMP_TO_DEDGE, BACKGROUND, REPEAT };

/* Continuous 3D grid interface
 *
 * Continuous grids allow interpolation between coordinates providing access to
 *floating point indices.
 */
template <typename T> class CGridInterface {
public:
  virtual ~CGridInterface() {}
  /* set
   * @i **[in]** coordinate (grid space)
   * @v **[in]** value
   */
  virtual void set(const ivec3 &i, const T &v) = 0;
  /* get
   * @i **[in]** coordinate (grid space)
   * @return value at coordinate **i**
   */
  virtual T operator()(const ivec3 &i) const = 0;
  virtual T &operator()(const ivec3 &i) = 0;
  /* get
   * @i **[in]** X coordinate (grid space)
   * @j **[in]** Y coordinate (grid space)
   * @k **[in]** Z coordinate (grid space)
   * @return value at coordinate **(i, j, k)**
   */
  virtual T operator()(const uint &i, const uint &j, const uint &k) const = 0;
  virtual T &operator()(const uint &i, const uint &j, const uint &k) = 0;
  /* get
   * @i **[in]** coordinate (grid space)
   * @return value at coordinate **i**
   */
  virtual T operator()(const vec3 &i) const = 0;
  /* get
   * @i **[in]** X coordinate (grid space)
   * @j **[in]** Y coordinate (grid space)
   * @k **[in]** Z coordinate (grid space)
   * @return value at coordinate **(i, j, k)**
   */
  virtual T operator()(const float &i, const float &j,
                       const float &k) const = 0;
  virtual void normalize() = 0;
  virtual void normalizeElements() = 0;
  /* test
   * @i coordinate
   *
   * @return **true** if the grid contains the coordinate **i**
   */
  bool belongs(const ivec3 &i) const { return ivec3() <= i && i < dimensions; }
  Point3 worldPosition(const ivec3 &ijk) const {
    return toWorld(Point3(ijk[0], ijk[1], ijk[2]));
  }
  // access mode
  CGridAccessMode mode;
  // default value
  T background;
  // grid dimensions
  ivec3 dimensions;
  // grid to world transform
  Transform toWorld;
  // world to grid transform
  Transform toGrid;
  T border;
  bool useBorder;
};

template <class T> class CGrid2DInterface {
public:
  void setTransform(const Transform2D &t) {
    toWorld = t;
    toGrid = inverse(t);
  }

  void setTransform(Vector2 _offset, Vector2 _cellSize) {
    offset = _offset;
    toWorld = scale(_cellSize.x, _cellSize.y) * translate(offset);
    toWorld.computeInverse();
    toGrid = inverse(toWorld);
  }

  virtual void setDimensions(uint32_t w, uint32_t h) {
    width = w;
    height = h;
  }

  void set(uint32_t w, uint32_t h, Vector2 _offset, Vector2 _cellSize) {
    setTransform(_offset, _cellSize);
    setDimensions(w, h);
  }

  Point<int, 2> cell(Point2 wp) const {
    vec2 delta(0.5f, 0.5f);
    return Point<int, 2>(toGrid(wp) + delta);
  }

  Point<int, 2> safeCell(Point2 wp) const {
    vec2 delta(0.5f, 0.5f);
    Point<int, 2> gp(toGrid(wp) + delta);
    gp[0] = std::min(static_cast<int>(width) - 1,
                     static_cast<int>(std::max(0, gp[0])));
    gp[1] = std::min(static_cast<int>(height) - 1,
                     static_cast<int>(std::max(0, gp[1])));
    return gp;
  }

  bool belongs(Point<int, 2> c) const {
    return 0 <= c[0] && c[0] < static_cast<int>(width) && 0 <= c[1] &&
           c[1] < static_cast<int>(height);
  }

  ponos::Point2 worldPosition(const ponos::ivec2 &ij) const {
    return toWorld(ponos::Point2(ij[0], ij[1]));
  }

  BBox2D cellWBox(const Point<int, 2> &c) const {
    Point2 pmin(c[0] - 0.5f, c[1] - 0.5f);
    Point2 pmax(c[0] + 0.5f, c[1] + 0.5f);
    return BBox2D(toWorld(pmin), toWorld(pmax));
  }

  ivec2 getDimensions() const { return ponos::ivec2(width, height); }

  T border;
  bool useBorder;

  uint32_t width, height;
  Vector2 offset;
  Transform2D toWorld, toGrid;
};

} // ponos namespace

#endif
