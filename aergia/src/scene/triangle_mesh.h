#ifndef AERGIA_SCENE_TRIANGLE_MESH
#define AERGIA_SCENE_TRIANGLE_MESH

#include "io/utils.h"
#include "scene/raw_mesh.h"
#include "scene/scene_object.h"

#include <memory>

namespace aergia {

	class TriangleMesh : SceneObject {
  	public:
			TriangleMesh(const std::string &filename);
	 		TriangleMesh(const RawMesh *m);
			virtual ~TriangleMesh() {}

			void draw() const override;
		protected:
			std::shared_ptr<const RawMesh> mesh;
	};

} // aergia namespace

#endif // AERGIA_SCENE_TRIANGLE_MESH

