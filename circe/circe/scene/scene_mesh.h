// Created by FilipeCN on 2/21/2018.
/*
 * Copyright (c) 2018 FilipeCN
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

#ifndef CIRCE_SCENE_MESH_H
#define CIRCE_SCENE_MESH_H

#include <circe/io/buffer.h>

namespace circe {

/// Set of buffers created that represent a raw mesh for rendering.
class SceneMesh {
public:
  /// \param rm raw mesh
  explicit SceneMesh(ponos::RawMesh &rm);
  /// Binds buffers
  void bind();
  void unbind();
  /// \return vertex buffer pointer
  const VertexBuffer *vertexBuffer() const;
  /// \return index buffer pointer
  const IndexBuffer *indexBuffer() const;

private:
  GLuint VAO;
  std::vector<float> vertexData_;
  std::vector<uint> indexData_;
  ponos::RawMesh &mesh_;
  VertexBuffer vertexBuffer_;
  IndexBuffer indexBuffer_;
};

} // circe namespace

#endif // CIRCE_SCENE_MESH_H
