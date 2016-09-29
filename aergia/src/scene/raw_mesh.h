#ifndef AERGIA_SCENE_RAW_MESH_H
#define AERGIA_SCENE_RAW_MESH_H

#include <ponos.h>

#include <vector>

namespace aergia {

	/* mesh structure
	 * Stores the elements of the mesh in simple arrays. This class
	 * is the one that actually stores the geometry, then other
	 * objects in the system can just use its reference avoiding
	 * duplicating data.
	 */
	class RawMesh {
  	public:
	 		RawMesh() {}
			virtual ~RawMesh() {}

			void apply(const ponos::Transform &t);

			void splitIndexData();

			void computeBBox();

			std::vector<float> vertices;
			std::vector<float> normals;
			std::vector<float> texcoords;
			struct IndexData {
				int vertexIndex;
				int normalIndex;
				int texcoordIndex;
			};
			std::vector<IndexData> indices;
			size_t vertexCount;
			std::vector<uint> verticesIndices;
			std::vector<uint> normalsIndices;
			std::vector<uint> texcoordsIndices;
			uint elementSize;
			ponos::BBox bbox;
	};

} // aergia namespace

#endif // AERGIA_SCENE_RAW_MESH_H

