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

#include <ponos/ponos.h>

#include <aergia/graphics/shader.h>
#include <aergia/io/buffer.h>
#include <aergia/io/utils.h>
#include <aergia/ui/interactive_object_interface.h>

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
    UNUSED_VARIABLE(t);
    UNUSED_VARIABLE(r);
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

  SceneMesh(ponos::RawMesh *m,
            std::function<void(Shader *s)> f =
            [](Shader *s) { UNUSED_VARIABLE(s); },
            Shader *s = nullptr) {
    this->visible = true;
    this->rawMesh = m;
    this->setupVertexBuffer();
    this->setupIndexBuffer();
    shader = s;
    drawCallback = f;
    if (shader)
      for (auto att : this->vb->bufferDescriptor.attributes)
        shader->addVertexAttribute(att.first.c_str());
  }

  virtual ~SceneMesh() {}

  void draw() const override {
    if (!visible)
      return;
    vb->bind();
    ib->bind();
    if (shader)
      shader->begin(this->vb.get());
    else {
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(
          0, vb->bufferDescriptor.elementSize, GL_FLOAT, GL_FALSE,
          0 /*sizeof(float) * vb->bufferDescriptor.elementSize*/, 0);
    }
    if (drawCallback)
      drawCallback(shader);
    glDrawElements(this->ib->bufferDescriptor.elementType,
                   this->ib->bufferDescriptor.elementCount, GL_UNSIGNED_INT, 0);
    if (shader)
      shader->end();
  }

  ponos::BBox getBBox() { return this->transform(rawMesh->bbox); }

  ponos::RawMesh *rawMesh;
  std::function<void(Shader *s)> drawCallback;

protected:
  virtual void setupVertexBuffer(GLuint _elementType = GL_TRIANGLES,
                                 GLuint _type = GL_ARRAY_BUFFER,
                                 GLuint _use = GL_STATIC_DRAW) {
    UNUSED_VARIABLE(_elementType);
    UNUSED_VARIABLE(_type);
    UNUSED_VARIABLE(_use);
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
        1, rawMesh->verticesIndices.size(), rawMesh->primitiveType);
    ib.reset(new IndexBuffer(&rawMesh->verticesIndices[0], indexDescriptor));
  }

  std::shared_ptr<VertexBuffer> vb;
  std::shared_ptr<IndexBuffer> ib;

private:
  Shader *shader;
};

typedef std::shared_ptr<SceneObject> SceneObjectSPtr;

} // aergia namespace

#endif // AERGIA_SCENE_SCENE_OBJECT_H
