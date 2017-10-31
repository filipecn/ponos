#ifndef AERGIA_SCENE_MESH_UTILS_H
#define AERGIA_SCENE_MESH_UTILS_H

#include <ponos.h>

namespace aergia {

	/* create
	 * @d **[in]** dimensions
	 * @s **[in]** scale
	 * @o **[in]** offset
	 * @return pointer to a <RawMesh> object with the list of vertices and indices that describe a grid in the 3D space.
	 */
	ponos::RawMesh *create_grid_mesh(const ponos::ivec3 &d, float s, const ponos::vec3 &o);

	/* create
	 * @m **[in]** base mesh
	 * return RawMesh representing the edges of **m**.
	 */
	ponos::RawMesh *create_wireframe_mesh(const ponos::RawMesh *m);

} // aergia namespace

#endif // AERGIA_SCENE_MESH_UTILS_H

