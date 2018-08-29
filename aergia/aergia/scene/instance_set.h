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

//
// Created by fuiri on 2/19/2018.
//

#ifndef AERGIA_INSTANCE_SET_H
#define AERGIA_INSTANCE_SET_H

#include <aergia/scene/scene_object.h>
#include <aergia/scene/scene_mesh.h>

namespace aergia {

/// Stores a base mesh and its instances descriptions on buffers for fast
/// rendering.
class InstanceSet : public SceneObject {
public:
  /// \param rm base mesh
  /// \param s shader
  /// \param n **[optional | def = 0]** number of instances
  explicit InstanceSet(SceneMesh &rm, ShaderProgram s, size_t n = 0);
  ~InstanceSet() override;
  /// \param d buffer descriptor
  /// \return index of the new buffer
  uint add(BufferDescriptor d);
  /// reserve memory for n instances
  /// \param n number of instances
  void resize(uint n);
  /// \param b buffer index (must belong to a float type)
  /// \param i instance index
  /// \return pointer to the first component of instance i
  float* instanceF(uint b, uint i);
  /// \param b buffer index (must belong to a uint type)
  /// \param i instance index
  /// \return pointer to the first component of instance i
  uint* instanceU(uint b, uint i);
  /// \param b buffer index
  void bind(uint b);
  void draw(const CameraInterface *camera, ponos::Transform transform) override;

private:
  ShaderProgram shader_; ///< instance shader program
  SceneMesh &baseMesh_; ///< base mesh
  VertexBuffer bv_; ///< base vertex buffer
  IndexBuffer bi_; ///< base index buffer
  uint count_; ///< number of instances
  std::vector<uint> buffersIndices_; ///< maps buffers -> data indices
  std::vector<bool> dataChanged_; ///< data of a buffer must be updated
  std::vector<std::vector<uint>> dataU_; ///< unsigned int data
  std::vector<std::vector<float>> dataF_; ///< float data
  std::vector<std::vector<uchar>> dataC_; ///< unsigned byte data
  std::vector<BufferInterface *> buffers_; ///< buffers
};

} // aergia namespace

#endif //AERGIA_INSTANCE_SET_H
