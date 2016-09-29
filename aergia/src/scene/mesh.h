#ifndef AERGIA_SCENE_MESH_H
#define AERGIA_SCENE_MESH_H

#include "scene/raw_mesh.h"

#include <ponos.h>

namespace aergia {

	/* mesh
	 * The pair of a Mesh and a transform.
	 */
	class Mesh {
		public:
			/* Constructor.
			 * @m **[in]**
			 * @t **[in]**
			 * @return
			 */
			Mesh(const RawMesh *m, const ponos::Transform &t);
			virtual ~Mesh() {}

			bool intersect(const ponos::Point3 &p);
			const ponos::BBox& getBBox() const;
			const RawMesh* getMesh() const;
			const ponos::Transform& getTransform() const;

		private:
			std::shared_ptr<const RawMesh> mesh;
			ponos::Transform transform;
			ponos::BBox bbox;
	};

} // aergia namespace

#endif // AERGIA_SCENE_MESH_H

