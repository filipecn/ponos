#ifndef AERGIA_HELPERS_WIREFRAME_MESH_H
#define AERGIA_HELPERS_WIREFRAME_MESH_H

#include "scene/scene_object.h"

#include <ponos.h>

namespace aergia {

	class WireframeMesh : public SceneMesh {
  	public:
			WireframeMesh(const std::string &filename);
	 		WireframeMesh(const ponos::RawMesh *m, const ponos::Transform &t);
			virtual ~WireframeMesh() {}
			/* @inherit */
			void draw() const override;

		protected:
			void setupIndexBuffer() override;
	};

} // aergia namespace

#endif // AERGIA_HELPERS_WIREFRAME_MESH_H

