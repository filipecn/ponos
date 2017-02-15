#ifndef AERGIA_SCENE_TRIANGLE_MESH
#define AERGIA_SCENE_TRIANGLE_MESH

#include "scene/scene_object.h"

#include <ponos.h>

#include <memory>

namespace aergia {

	class TriangleMesh : public SceneMesh {
  	public:
			TriangleMesh(const std::string &filename);
	 		TriangleMesh(const ponos::RawMesh *m);
			virtual ~TriangleMesh() {}
			/* @inherit */
			void draw() const override;
	};

} // aergia namespace

#endif // AERGIA_SCENE_TRIANGLE_MESH

