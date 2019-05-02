/*
 * Copyright (c) 2019 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * iM the Software without restriction, including without limitation the rights
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

#ifndef HERMES_NUMERIC_CUDA_VECTOR_GRID_H
#define HERMES_NUMERIC_CUDA_VECTOR_GRID_H

#include <hermes/numeric/cuda_grid.h>

namespace hermes {

namespace cuda {

class VectorGridTexture2 {
public:
  VectorGridTexture2() {
    uGrid.setOrigin(point2f(0, 0));
    vGrid.setOrigin(point2f(0, 0));
  }
  /// \param resolution in number of cells
  /// \param origin (0,0) corner position
  /// \param dx cell size
  VectorGridTexture2(vec2u resolution, point2f origin, float dx) {
    uGrid.resize(resolution);
    uGrid.setOrigin(origin);
    uGrid.setDx(dx);
    vGrid.resize(resolution);
    vGrid.setOrigin(origin);
    vGrid.setDx(dx);
  }
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  virtual void resize(vec2u res) {
    uGrid.resize(res);
    vGrid.resize(res);
  }
  /// Changes grid origin position
  /// \param o in world space
  virtual void setOrigin(const point2f &o) {
    uGrid.setOrigin(o);
    vGrid.setOrigin(o);
  }
  /// Changes grid cell size
  /// \param d new size
  virtual void setDx(float d) {
    uGrid.setDx(d);
    vGrid.setDx(d);
  }
  virtual void copy(const VectorGridTexture2 &other) {
    uGrid.copy(other.uGrid);
    vGrid.copy(other.vGrid);
  }
  virtual float *uDeviceData() { return uGrid.texture().deviceData(); }
  virtual float *vDeviceData() { return vGrid.texture().deviceData(); }
  virtual const float *uDeviceData() const {
    return uGrid.texture().deviceData();
  }
  virtual const float *vDeviceData() const {
    return vGrid.texture().deviceData();
  }
  virtual const GridTexture2<float> &u() const { return uGrid; }
  virtual const GridTexture2<float> &v() const { return vGrid; }
  virtual GridTexture2<float> &u() { return uGrid; }
  virtual GridTexture2<float> &v() { return vGrid; }

protected:
  GridTexture2<float> uGrid, vGrid;
};

} // namespace cuda

} // namespace hermes

#endif // HERMES_STRUCTURES_CUDA_VECTOR_GRID_H