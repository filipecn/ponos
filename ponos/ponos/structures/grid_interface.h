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

#ifndef PONOS_STRUCTURES_C_GRID_INTERFACE_H
#define PONOS_STRUCTURES_C_GRID_INTERFACE_H

#include <ponos/geometry/bbox.h>
#include <ponos/geometry/transform.h>

#include <algorithm>
#include <functional>

namespace ponos {

enum class CGridAccessMode { CLAMP_TO_DEDGE, BACKGROUND, REPEAT };
enum class GridDataPosition {
  CELL_CENTER,
  VERTEX_CENTER,
  U_FACE_CENTER,
  V_FACE_CENTER,
  W_FACE_CENTER
};
enum class GridAccessMode { CLAMP_TO_EDGE, BORDER, REPEAT, RESTRICT };

/** \brief Continuous 3D grid interface
 *
 * Continuous grids allow interpolation between coordinates providing access to
 *floating point indices.
 */
template <typename T> class CGridInterface {
public:
  virtual ~CGridInterface() {}
  /* set
   * \param i **[in]** coordinate (grid space)
   * \param v **[in]** value
   */
  virtual void set(const ivec3 &i, const T &v) = 0;
  /* get
   * \param i **[in]** coordinate (grid space)
   * \return value at coordinate **i**
   */
  virtual T operator()(const ivec3 &i) const = 0;
  /* get
   * \param i **[in]** coordinate (grid space)
   * \return value at coordinate **i**
   */
  virtual T &operator()(const ivec3 &i) = 0;
  /* get
   * \param i **[in]** X coordinate (grid space)
   * \param j **[in]** Y coordinate (grid space)
   * \param k **[in]** Z coordinate (grid space)
   * \return value at coordinate **(i, j, k)**
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

/** \brief 2D data grid interface
 *
 * Continuous grids allow interpolation between coordinates providing access to
 *floating point indices.
 */
template <typename T> class Grid2DInterface {
public:
  typedef T DataType; //!< grid data typename
  virtual T getData(int i, int j) const = 0;
  virtual T &getData(int i, int j) = 0;
  virtual void updateDataStructure() = 0;
  /** iterate through data elements
   * \param f function called on each element
   */
  void forEach(const std::function<void(T &d, size_t i, size_t j)> &f) {
    for (size_t i = 0; i < width; i++)
      for (size_t j = 0; j < height; j++)
        f(getData(i, j), i, j);
  }
  /** iterate through data elements
   * \param f function called on each element
   */
  void
  forEach(const std::function<void(const T &d, size_t i, size_t j)> &f) const {
    for (size_t i = 0; i < width; i++)
      for (size_t j = 0; j < height; j++)
        f(getData(i, j), i, j);
  }
  T accessData(int i, int j) const {
    if (!belongs(Point<int, 2>({i, j}))) {
      switch (accessMode) {
      case GridAccessMode::BORDER:
        return border;
      case GridAccessMode::CLAMP_TO_EDGE:
        return safeData(i, j);
      case GridAccessMode::RESTRICT:
        throw "restrict access to grid!";
      default:
        return border;
      }
    }
    return getData(i, j);
  }
  T &accessData(int i, int j) {
    if (!belongs(Point<int, 2>({i, j}))) {
      switch (accessMode) {
      case GridAccessMode::BORDER:
        return border;
      case GridAccessMode::CLAMP_TO_EDGE:
        return safeData(i, j);
      case GridAccessMode::RESTRICT:
        throw "restrict access to grid!";
      default:
        return border;
      }
    }
    return getData(i, j);
  }
  T &operator()(const ivec2 &ij) { return accessData(ij[0], ij[1]); }
  T operator()(const ivec2 &ij) const { return accessData(ij[0], ij[1]); }
  T &operator()(int i, int j) { return accessData(i, j); }
  T operator()(int i, int j) const { return accessData(i, j); }
  T safeData(int i, int j) const {
    return getData(std::max(0, std::min(static_cast<int>(width) - 1, i)),
                   std::max(0, std::min(static_cast<int>(height) - 1, j)));
  }
  T &safeData(int i, int j) {
    return getData(std::max(0, std::min(static_cast<int>(width) - 1, i)),
                   std::max(0, std::min(static_cast<int>(height) - 1, j)));
  }
  T dSample(float x, float y, T r) const {
    ponos::Point<int, 2> gp = this->cell(Point2(x, y));
    if (!this->belongs(gp))
      return r;
    return getData(gp[0], gp[1]);
  }
  virtual void setAll(const T t) {
    for (uint32_t i = 0; i < width; i++)
      for (uint32_t j = 0; j < height; ++j)
        getData(i, j) = t;
  }
  /** Sets grid's transform
   * \param t transformation
   */
  void setTransform(const Transform2D &t) {
    toWorld = t;
    toGrid = inverse(t);
  }
  /** Sets grid's transform
   * \param _offset **[in]** origin
   * \param _cellSize **[in]** grid spacing
   */
  void setTransform(Vector2 _offset, Vector2 _cellSize) {
    offset = _offset;
    toWorld = scale(_cellSize.x, _cellSize.y) * translate(offset);
    toWorld.computeInverse();
    toGrid = inverse(toWorld);
  }
  /** Sets grid's dimensions
   * \param w width **[in]** (number of cells)
   * \param h height **[in]** (number of cells)
   */
  void setDimensions(uint32_t w, uint32_t h) {
    width = w;
    height = h;
    updateDataStructure();
  }
  /** Sets grid's dimensions
   * \param d (width,height) **[in]** (number of cells)
   */
  void set(const ivec2 &d) { setDimensions(d[0], d[1]); }
  /** Sets grid's properties
   * \param w width **[in]** (number of cells)
   * \param h height **[in]** (number of cells)
   * \param _offset **[in]** origin
   * \param _cellSize **[in]** grid spacing
   */
  void set(uint32_t w, uint32_t h, Vector2 _offset, Vector2 _cellSize) {
    setTransform(_offset, _cellSize);
    setDimensions(w, h);
  }
  /** Fit grid to bounding box
   * \param w width **[in]** (number of cells)
   * \param h height **[in]** (number of cells)
   * \param b bounding box
   */
  void set(uint32_t w, uint32_t h, const BBox2D &b) {
    setDimensions(w, h);
    vec2 size(w, h);
    setTransform(translate(vec2(b.pMin)) *
                 scale(b.size(0) / size.x, b.size(1) / size.y));
  }
  /** data position
   * \return grid data position relative to grid origin
   */
  vec2 dataOffset() const {
    switch (dataPosition) {
    case GridDataPosition::CELL_CENTER:
      return vec2(0.5f, 0.5f);
    case GridDataPosition::VERTEX_CENTER:
      return vec2(0.f, 0.f);
    case GridDataPosition::U_FACE_CENTER:
      return vec2(0.f, 0.5f);
    case GridDataPosition::V_FACE_CENTER:
      return vec2(0.5f, 0.f);
    default:
      FATAL_ASSERT(false);
    }
    return vec2();
  }
  /** Cell containing point
   * \param wp point (world position)
   * \return point representing cell's position (index space)
   */
  Point<int, 2> cell(Point2 wp) const { return Point<int, 2>(toGrid(wp)); }
  /** Data cell containing point
   * \param wp point (world position)
   * \return point representing cell's position (index space)
   */
  Point<int, 2> dataCell(Point2 wp) const {
    return Point<int, 2>(toGrid(wp) - dataOffset());
  }
  /** Interior cell containing point
   * \param wp **[in]** point (world position)
   * \return point representing cell's position (index space)
   */
  Point<int, 2> safeCell(Point2 wp) const {
    Point<int, 2> gp = cell(wp);
    gp[0] = std::min(static_cast<int>(width) - 1,
                     static_cast<int>(std::max(0, gp[0])));
    gp[1] = std::min(static_cast<int>(height) - 1,
                     static_cast<int>(std::max(0, gp[1])));
    return gp;
  }
  /** Checks if cell belongs to grid
   * \param c **[in]** cell (index space)
   * \returns **true** if cell belongs to grid
   */
  bool belongs(Point<int, 2> c) const {
    return 0 <= c[0] && c[0] < static_cast<int>(width) && 0 <= c[1] &&
           c[1] < static_cast<int>(height);
  }
  /** World space to data grid space (diff from index space)
   * \param wp **[in]** world space
   * \return data grid position
   */
  Point2 dataGridPosition(const Point2 &wp) const {
    return toGrid(wp) - dataOffset();
  }
  /** Index space to world position of data
   * \param ij **[in]** index
   * \return world position
   */
  Point2 dataWorldPosition(const ivec2 &ij) const {
    return toWorld(ponos::Point2(ij[0], ij[1]) + dataOffset());
  }
  /** Index space to world position of data
   * \param i **[in]** index in x direction
   * \param j **[in]** index in y direction
   * \return world position
   */
  Point2 dataWorldPosition(int i, int j) const {
    return toWorld(ponos::Point2(i, j) + dataOffset());
  }
  /** Bounding box of cell **c**
   * \param c **[in]** cell (index space)
   * \return bounding box of **c** in world coordinates
   */
  BBox2D cellWBox(const Point<int, 2> &c) const {
    Point2 pmin(c[0], c[1]);
    Point2 pmax(c[0] + 1, c[1] + 1);
    return BBox2D(toWorld(pmin), toWorld(pmax));
  }
  /** Get grid spacing
   * \return cell size
   */
  ivec2 getDimensions() const { return ivec2(width, height); }
  /** Get grid dimensions
   * \return dimensions
   */
  vec2 cellSize() const {
    BBox2D b = this->cellWBox(Point<int, 2>({0, 0}));
    return b.extends();
  }
  T border;                      //!< border value
  uint32_t width;                //!< number of cells in x direction
  uint32_t height;               //!< number of cells in y direction
  Vector2 offset;                //!< grid origin
  Transform2D toWorld;           //!< grid space to world space transform
  Transform2D toGrid;            //!< world space to grid space transform
  GridDataPosition dataPosition; //!< grid data position type
  GridAccessMode accessMode;     //!< grid access mode
};

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const Grid2DInterface<T> *g) {
  for (size_t j = 0; j < g->height; j++) {
    for (size_t i = 0; i < g->width; i++)
      os << (*g)(i, j) << " ";
    os << std::endl;
  }
  os << std::endl;
  return os;
}
}

#endif // PONOS_STRUCTURES_C_GRID_INTERFACE_H
