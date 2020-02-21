/*
 * Copyright (c) 2019 FilipeCN
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

#ifndef HERMES_NUMERIC_FIELD_H
#define HERMES_NUMERIC_FIELD_H

#include <hermes/geometry/transform.h>
#include <hermes/storage/cuda_texture.h>

namespace hermes {

namespace cuda {

/// Stores grid data into a texture for fast read from kernel code and also in
/// global memory for write calls.
/// \tparam T data type
template <typename T> class FieldTexture2 {
public:
  FieldTexture2() = default;
  /// \param toField world to field transform
  /// \param size texture resolution
  FieldTexture2(const Transform2<float> toField, const size2 &size)
      : toField(toField), toWorld(inverse(toField)) {
    texture_.resize(size[0], size[1]);
  }
  /// \param toFieldTransform world to field transform
  void setTransform(const Transform2<float> toWorldTransform) {
    toWorld = toWorldTransform;
    toField = inverse(toWorldTransform);
  }
  size2 resolution() const {
    return size2(texture_.width(), texture_.height());
  }
  /// \param size texture resolution
  void resize(const size2 &size) { texture_.resize(size[0], size[1]); }
  /// \return Texture<T>& reference to texture object
  Texture<T> &texture() { return texture_; }
  /// \return const Texture<T>& const reference to texture object
  const Texture<T> &texture() const { return texture_; }
  /// \return const Transform<float, D>& world to field transform
  const Transform2<float> &toFieldTransform() const { return toField; }
  /// \return const Transform<float, D>& field to world transform
  const Transform2<float> &toWorldTransform() const { return toWorld; }

private:
  Transform2<float> toField;
  Transform2<float> toWorld;
  Texture<T> texture_;
};

template <typename T> class FieldTexture3 {
public:
  FieldTexture3() = default;
  /// \param toField world to field transform
  /// \param size texture resolution
  FieldTexture3(const Transform<float> toField, const vec3u &size)
      : toField(toField), toWorld(inverse(toField)) {
    texture_.resize(size[0], size[1], size[2]);
  }
  /// \param toFieldTransform world to field transform
  void setTransform(const Transform<float> toWorldTransform) {
    toWorld = toWorldTransform;
    toField = inverse(toWorldTransform);
  }
  vec3u resolution() const {
    return vec3u(texture_.width(), texture_.height(), texture_.depth());
  }
  /// \param size texture resolution
  void resize(const vec3u &size) { texture_.resize(size[0], size[1], size[2]); }
  /// \return Texture<T>& reference to texture object
  Texture3<T> &texture() { return texture_; }
  /// \return const Texture<T>& const reference to texture object
  const Texture3<T> &texture() const { return texture_; }
  /// \return const Transform<float, D>& world to field transform
  const Transform<float> &toFieldTransform() const { return toField; }
  /// \return const Transform<float, D>& field to world transform
  const Transform<float> &toWorldTransform() const { return toWorld; }

private:
  Transform<float> toField;
  Transform<float> toWorld;
  Texture3<T> texture_;
};

} // namespace cuda

} // namespace hermes

#endif // HERMES_NUMERIC_FIELD_H