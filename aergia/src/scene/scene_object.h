#ifndef AERGIA_SCENE_SCENE_OBJECT_H
#define AERGIA_SCENE_SCENE_OBJECT_H

#include <ponos.h>

#include "io/buffer.h"
#include "io/utils.h"
#include "scene/raw_mesh.h"

namespace aergia {

	/* interface
	 * The Scene Object Interface represents an drawable object that can be intersected
	 * by a ray.
	 */
	class SceneObject {
  	public:
	 		SceneObject()
			: selected(false), visible(true) {}
			virtual ~SceneObject() {}

			/* draw
			 * render method
			 */
			virtual void draw() const = 0;
			/* query
			 * @r **[in]** ray
			 * @t **[out]** receives the parametric value of the intersection
			 * @return **true** if intersection is found
			 */
			virtual bool intersect(const ponos::Ray3 &r, float *t = nullptr) const { return false; }

			ponos::Transform transform;
			bool selected;
			bool visible;
	};

	class SceneMesh : public SceneObject {
		public:
			SceneMesh() {}
			SceneMesh(const std::string &filename) {
				RawMesh *m = new RawMesh();
				loadOBJ(filename, m);
				m->computeBBox();
				m->splitIndexData();
				mesh.reset(m);
				setupVertexBuffer();
				setupIndexBuffer();
			}
			SceneMesh(const RawMesh *m) {
				mesh.reset(m);
				setupVertexBuffer();
				setupIndexBuffer();
			}
			virtual ~SceneMesh() {}

			void draw() const override {
				if(!visible)
					return;
				vb->bind();
				ib->bind();
			}

			std::shared_ptr<const RawMesh> mesh;

		protected:
			virtual void setupVertexBuffer() {
				BufferDescriptor vertexDescriptor(3, mesh->vertexCount);
				vb.reset(new VertexBuffer(&mesh->vertices[0], vertexDescriptor));
			}

			virtual void setupIndexBuffer() {
				BufferDescriptor indexDescriptor = create_index_buffer_descriptor(1, mesh->verticesIndices.size());
				ib.reset(new IndexBuffer(&mesh->verticesIndices[0], indexDescriptor));
			}

			std::shared_ptr<const VertexBuffer> vb;
			std::shared_ptr<const IndexBuffer> ib;
	};

} // aergia namespace

#endif // AERGIA_SCENE_SCENE_OBJECT_H
