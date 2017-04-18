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

#ifndef AERGIA_SCENE_SCENE_OBJECT_H
#define AERGIA_SCENE_SCENE_OBJECT_H

#include <ponos.h>

#include "io/buffer.h"
#include "io/utils.h"
#include "ui/interactive_object_interface.h"

namespace aergia {

/** \brief interface
 * The Scene Object Interface represents an drawable object that can be
 * intersected
 * by a ray.
 */
class SceneObject : public InteractiveObjectInterface {
public:
  SceneObject() : visible(true) {}
  virtual ~SceneObject() {}

  /** \brief draw
   * render method
   */
  virtual void draw() const = 0;
  /** \brief query
   * \param r **[in]** ray
   * \param t **[out]** receives the parametric value of the intersection
   * \return **true** if intersection is found
   */
  virtual bool intersect(const ponos::Ray3 &r, float *t = nullptr) {
    return false;
  }

  void updateTransform() override {
    transform = this->trackball.tb.transform * transform;
  }

  ponos::Transform transform;
  bool visible;
};

class SceneMesh : public SceneObject {
public:
  SceneMesh() {}
  SceneMesh(const std::string &filename) {
    ponos::RawMesh *m = new ponos::RawMesh();
    loadOBJ(filename, m);
    m->computeBBox();
    m->splitIndexData();
    m->buildInterleavedData();
    rawMesh = m;
    setupVertexBuffer();
    setupIndexBuffer();
  }
  SceneMesh(const ponos::RawMesh *m) {
    rawMesh = m;
    setupVertexBuffer();
    setupIndexBuffer();
  }
  virtual ~SceneMesh() {}

  void draw() const override {
    if (!visible)
      return;
    vb->bind();
    ib->bind();
  }

  ponos::BBox getBBox() { return this->transform(rawMesh->bbox); }

  const ponos::RawMesh *rawMesh;

protected:
  virtual void setupVertexBuffer(GLuint _elementType = GL_TRIANGLES,
                                 GLuint _type = GL_ARRAY_BUFFER,
                                 GLuint _use = GL_STATIC_DRAW) {
    BufferDescriptor dataDescriptor(rawMesh->interleavedDescriptor.elementSize,
                                    rawMesh->interleavedDescriptor.count);
    dataDescriptor.addAttribute(std::string("position"),
                                rawMesh->vertexDescriptor.elementSize, 0,
                                GL_FLOAT);
    size_t offset = rawMesh->vertexDescriptor.elementSize;
    if (rawMesh->normalDescriptor.count) {
      dataDescriptor.addAttribute(std::string("normal"),
                                  rawMesh->normalDescriptor.elementSize,
                                  offset * sizeof(float), GL_FLOAT);
      offset += rawMesh->normalDescriptor.elementSize;
    }
    if (rawMesh->texcoordDescriptor.count) {
      dataDescriptor.addAttribute(std::string("texcoord"),
                                  rawMesh->texcoordDescriptor.elementSize,
                                  offset * sizeof(float), GL_FLOAT);
      offset += rawMesh->texcoordDescriptor.elementSize;
    }
    vb.reset(new VertexBuffer(&rawMesh->interleavedData[0], dataDescriptor));
  }

  virtual void setupIndexBuffer() {
    BufferDescriptor indexDescriptor = create_index_buffer_descriptor(
        1, rawMesh->verticesIndices.size(),
        (rawMesh->meshDescriptor.elementSize == 4) ? GL_QUADS : GL_TRIANGLES);
    ib.reset(new IndexBuffer(&rawMesh->verticesIndices[0], indexDescriptor));
  }

  std::shared_ptr<const VertexBuffer> vb;
  std::shared_ptr<const IndexBuffer> ib;
};

class DynamicSceneMesh : public SceneObject {
public:
  DynamicSceneMesh() {}
  DynamicSceneMesh(const std::string &filename) {
    ponos::RawMesh *m = new ponos::RawMesh();
    loadOBJ(filename, m);
    m->computeBBox();
    m->splitIndexData();
    m->buildInterleavedData();
    rawMesh = m;
    setupVertexBuffer();
    setupIndexBuffer();
  }
  DynamicSceneMesh(ponos::RawMesh *m) {
    rawMesh = m;
    setupVertexBuffer();
    setupIndexBuffer();
  }
  virtual ~DynamicSceneMesh() {}

  void draw() const override {
    if (!visible)
      return;
    vb->bind();
    ib->bind();
  }

  ponos::BBox getBBox() { return this->transform(rawMesh->bbox); }

  ponos::RawMesh *rawMesh;

protected:
  virtual void setupVertexBuffer(GLuint _elementType = GL_TRIANGLES,
                                 GLuint _type = GL_ARRAY_BUFFER,
                                 GLuint _use = GL_STATIC_DRAW) {
    BufferDescriptor dataDescriptor(rawMesh->interleavedDescriptor.elementSize,
                                    rawMesh->interleavedDescriptor.count);
    dataDescriptor.addAttribute(std::string("position"),
                                rawMesh->vertexDescriptor.elementSize, 0,
                                GL_FLOAT);
    size_t offset = rawMesh->vertexDescriptor.elementSize;
    if (rawMesh->normalDescriptor.count) {
      dataDescriptor.addAttribute(std::string("normal"),
                                  rawMesh->normalDescriptor.elementSize,
                                  offset * sizeof(float), GL_FLOAT);
      offset += rawMesh->normalDescriptor.elementSize;
    }
    if (rawMesh->texcoordDescriptor.count) {
      dataDescriptor.addAttribute(std::string("texcoord"),
                                  rawMesh->texcoordDescriptor.elementSize,
                                  offset * sizeof(float), GL_FLOAT);
      offset += rawMesh->texcoordDescriptor.elementSize;
    }
    vb.reset(new VertexBuffer(&rawMesh->interleavedData[0], dataDescriptor));
  }

  virtual void setupIndexBuffer() {
    BufferDescriptor indexDescriptor = create_index_buffer_descriptor(
        1, rawMesh->verticesIndices.size(),
        (rawMesh->meshDescriptor.elementSize == 4) ? GL_QUADS : GL_TRIANGLES);
    ib.reset(new IndexBuffer(&rawMesh->verticesIndices[0], indexDescriptor));
  }

  std::shared_ptr<VertexBuffer> vb;
  std::shared_ptr<IndexBuffer> ib;
};

} // aergia namespace

#endif // AERGIA_SCENE_SCENE_OBJECT_H
