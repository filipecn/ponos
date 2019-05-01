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

#ifndef HERMES_NUMERIC_CUDA_GRID_H
#define HERMES_NUMERIC_CUDA_GRID_H

#include <hermes/numeric/cuda_field.h>

namespace hermes {

namespace cuda {

struct Grid2Info {
  Transform2<float> toField;
  Transform2<float> toWorld;
  vec2u resolution;
  point2f origin;
  float dx;
};

/// Represents a texture field with position offset and scale
template <typename T> class GridTexture2 {
public:
  GridTexture2() = default;
  /// \param resolution in number of cells
  /// \param origin (0,0) coordinate position
  /// \param dx cell size
  GridTexture2(vec2u resolution, point2f origin, float dx)
      : origin(origin), dx(dx) {
    texGrid.resize(resolution);
    updateTransform();
  }
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  void resize(vec2u res) { texGrid.resize(res); }
  /// Changes grid origin position
  /// \param o in world space
  void setOrigin(const point2f &o) {
    origin = o;
    updateTransform();
  }
  /// Changes grid cell size
  /// \param d new size
  void setDx(float d) {
    dx = d;
    updateTransform();
  }
  T minValue() {}
  /// \return Info grid info to be passed to kernels
  Grid2Info info() const {
    return {texGrid.toFieldTransform(), texGrid.toWorldTransform(),
            vec2u(texGrid.texture().width(), texGrid.texture().height()),
            origin, dx};
  }
  void copy(const GridTexture2 &other) {
    origin = other.origin;
    dx = other.dx;
    texGrid.texture().copy(other.texGrid.texture());
  }
  ///
  /// \return Texture<T>&
  Texture<T> &texture() { return texGrid.texture(); }
  ///
  /// \return const Texture<T>&
  const Texture<T> &texture() const { return texGrid.texture(); }
  ///
  /// \return Transform<float, 2>
  Transform2<float> toWorldTransform() const {
    return texGrid.toWorldTransform();
  }
  ///
  /// \return Transform2<float>
  Transform2<float> toFieldTransform() const {
    return texGrid.toFieldTransform();
  }

private:
  void updateTransform() {
    texGrid.setTransform(scale(dx, dx) *
                         translate(vec2f(origin[0], origin[1])));
    //  *
    //  scale(dx, dx));
  }
  point2f origin;
  float dx = 1.f;
  FieldTexture2<T> texGrid;
};

} // namespace cuda

} // namespace hermes

#endif // HERMES_STRUCTURES_CUDA_GRID_H