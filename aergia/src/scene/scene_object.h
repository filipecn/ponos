#ifndef AERGIA_SCENE_SCENE_OBJECT_H
#define AERGIA_SCENE_SCENE_OBJECT_H

#include <ponos.h>

#include "io/buffer.h"
#include "io/utils.h"
#include "ui/interactive_object_interface.h"

namespace aergia {

/* interface
 * The Scene Object Interface represents an drawable object that can be
 * intersected
 * by a ray.
 */
class SceneObject : public InteractiveObjectInterface {
public:
  SceneObject() : visible(true) {}
  virtual ~SceneObject() {}

  /* draw
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
    rawMesh.reset(m);
    setupVertexBuffer();
    setupIndexBuffer();
  }
  SceneMesh(const ponos::RawMesh *m) {
    rawMesh.reset(m);
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

  std::shared_ptr<const ponos::RawMesh> rawMesh;

protected:
  virtual void setupVertexBuffer() {
    BufferDescriptor vertexDescriptor(3, rawMesh->vertexCount);
    vb.reset(new VertexBuffer(&rawMesh->vertices[0], vertexDescriptor));
  }

  virtual void setupIndexBuffer() {
    BufferDescriptor indexDescriptor =
        create_index_buffer_descriptor(1, rawMesh->verticesIndices.size());
    ib.reset(new IndexBuffer(&rawMesh->verticesIndices[0], indexDescriptor));
  }

  std::shared_ptr<const VertexBuffer> vb;
  std::shared_ptr<const IndexBuffer> ib;
};

} // aergia namespace

#endif // AERGIA_SCENE_SCENE_OBJECT_H
