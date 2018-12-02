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

#ifndef CIRCE_SCENE_SCENE_OBJECT_H
#define CIRCE_SCENE_SCENE_OBJECT_H

#include <ponos/ponos.h>

#include <circe/graphics/shader.h>
#include <circe/io/buffer.h>
#include <circe/io/utils.h>
#include <circe/scene/scene_mesh.h>
#include <circe/ui/interactive_object_interface.h>

namespace circe {

/** \brief interface
 * The Scene Object Interface represents an drawable object that can be
 * intersected
 * by a ray.
 */
class SceneObject : public InteractiveObjectInterface {
public:
  SceneObject() : visible(true) {}
  virtual ~SceneObject() {}
  /// render method
  /// \param c cur active camera
  /// \param t object transform
  virtual void draw(const CameraInterface *c, ponos::Transform t) = 0;
  /// query
  /// \param r **[in]** ray
  /// \param t **[out]** receives the parametric value of the intersection
  /// \return **true** if intersection is found
  virtual bool intersect(const ponos::Ray3 &r, float *t = nullptr) {
    UNUSED_VARIABLE(t);
    UNUSED_VARIABLE(r);
    return false;
  }

  void updateTransform() override {
    transform = this->trackball.tb.getTransform() * transform;
  }

  ponos::Transform transform;
  bool visible;
};

class SceneMeshObject : public SceneObject {
public:
  SceneMeshObject() {}

  SceneMeshObject(const std::string &filename) {
    ponos::RawMeshSPtr rawMesh(new ponos::RawMesh());
    loadOBJ(filename, rawMesh.get());
    rawMesh->computeBBox();
    rawMesh->splitIndexData();
    rawMesh->buildInterleavedData();
    mesh_ = createSceneMeshPtr(rawMesh);
  }

  SceneMeshObject(const std::string &filename, ShaderProgramPtr s)
      : SceneMeshObject(filename) {
    shader_ = s;
  }

  SceneMeshObject(ponos::RawMesh *m,
                  std::function<void(ShaderProgram *, const CameraInterface *,
                                     ponos::Transform)>
                      f =
                          [](ShaderProgram *s, const CameraInterface *camera,
                             ponos::Transform t) {
                            UNUSED_VARIABLE(s);
                            UNUSED_VARIABLE(camera);
                            UNUSED_VARIABLE(t);
                          },
                  ShaderProgram *s = nullptr) {
    this->visible = true;
    shader_.reset(s);
    drawCallback = f;
    // if (shader)
    //  for (auto att : this->vb->bufferDescriptor.attributes)
    //    shader->addVertexAttribute(att.first.c_str());
  }

  virtual ~SceneMeshObject() {}

  void draw(const CameraInterface *camera, ponos::Transform t) override {
    if (!visible)
      return;
    mesh_->bind();
    if (shader_) {
      mesh_->vertexBuffer()->locateAttributes(*shader_.get());
      shader_->begin();
    } else {
      glEnableVertexAttribArray(0);
      // glVertexAttribPointer(
      // 0, vb->bufferDescriptor.elementSize, GL_FLOAT, GL_FALSE,
      // 0 /*sizeof(float) * vb->bufferDescriptor.elementSize*/, 0);
    }
    if (drawCallback)
      drawCallback(shader_.get(), camera, transform * t);
    glDrawElements(mesh_->indexBuffer()->bufferDescriptor.elementType,
                   mesh_->indexBuffer()->bufferDescriptor.elementCount *
                       mesh_->indexBuffer()->bufferDescriptor.elementSize,
                   GL_UNSIGNED_INT, 0);
    mesh_->unbind();
    if (shader_)
      shader_->end();
  }

  void setShader(ShaderProgramPtr shader) { shader_ = shader; }
  ShaderProgramPtr shader() { return shader_; }
  SceneMeshPtr mesh() { return mesh_; }

  // ponos::BBox getBBox() { return this->transform(rawMesh->bbox); }

  std::function<void(ShaderProgram *, const CameraInterface *,
                     ponos::Transform)>
      drawCallback;

protected:
  /*virtual void setupVertexBuffer(GLuint _elementType = GL_TRIANGLES,
                                 GLuint _type = GL_ARRAY_BUFFER,
                                 GLuint _use = GL_STATIC_DRAW) {
    UNUSED_VARIABLE(_elementType);
    UNUSED_VARIABLE(_type);
    UNUSED_VARIABLE(_use);
    BufferDescriptor dataDescriptor(rawMesh->interleavedDescriptor.elementSize,
                                    rawMesh->interleavedDescriptor.count);
    dataDescriptor.addAttribute(std::string("position"),
                                rawMesh->positionDescriptor.elementSize, 0,
                                GL_FLOAT);
    size_t offset = rawMesh->positionDescriptor.elementSize;
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
        1, rawMesh->positionsIndices.size(), rawMesh->primitiveType);
    ib.reset(new IndexBuffer(&rawMesh->positionsIndices[0], indexDescriptor));
  }

  std::shared_ptr<VertexBuffer> vb;
  std::shared_ptr<IndexBuffer> ib;
*/
  SceneMeshPtr mesh_;
  ShaderProgramPtr shader_;
};

typedef std::shared_ptr<SceneObject> SceneObjectSPtr;
typedef std::shared_ptr<SceneMeshObject> SceneMeshObjectSPtr;

template <typename... TArg>
SceneMeshObjectSPtr createSceneMeshObjectSPtr(TArg &&... Args) {
  return std::make_shared<SceneMeshObject>(std::forward<TArg>(Args)...);
}

} // namespace circe

#endif // CIRCE_SCENE_SCENE_OBJECT_H
