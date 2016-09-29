#include "scene/wireframe_mesh.h"

namespace aergia {

	WireframeMesh::WireframeMesh(const std::string &filename)
	: SceneMesh(filename) {}


	WireframeMesh::WireframeMesh(const RawMesh *m, const ponos::Transform &t) {
		mesh.reset(m);
		setupVertexBuffer();
		setupIndexBuffer();
		transform = t;
	}

	void WireframeMesh::draw() const {
		glPushMatrix();
		vb->bind();
		ib->bind();
		float pm[16];
		transform.matrix().column_major(pm);
		glMultMatrixf(pm);
		glColor4f(0, 0, 0, 0.1);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, 0);
		glDrawElements(GL_LINES, ib->bufferDescriptor.elementCount, GL_UNSIGNED_INT, 0);
		glPopMatrix();
	}

	void WireframeMesh::setupIndexBuffer() {
		BufferDescriptor indexDescriptor = create_index_buffer_descriptor(1, mesh->verticesIndices.size(), GL_LINES);
		ib.reset(new IndexBuffer(&mesh->verticesIndices[0], indexDescriptor));
	}

} // aergia namespace
