#ifndef PONOS_STRUCTURES_RAW_MESH_H
#define PONOS_STRUCTURES_RAW_MESH_H

#include "geometry/bbox.h"
#include "geometry/transform.h"

#include <vector>

namespace ponos {

	/* mesh structure
	 * Stores the elements of the mesh in simple arrays. This class
	 * is the one that actually stores the geometry, then other
	 * objects in the system can just use its reference avoiding
	 * duplicating data.
	 */
	class RawMesh {
  	public:
	 		RawMesh() { dimensions = 3; }
			virtual ~RawMesh() {}

			void apply(const Transform &t);

			void splitIndexData();

			void computeBBox();

			ponos::BBox elementBBox(size_t i) const;
			ponos::Point3 vertexElement(size_t e, size_t v) const;

			struct IndexData {
				// index space
				int vertexIndex;
				int normalIndex;
				int texcoordIndex;
			};
			std::vector<float> vertices;
			std::vector<float> normals;
			std::vector<float> texcoords;
			std::vector<IndexData> indices;
			std::vector<uint> verticesIndices;
			std::vector<uint> normalsIndices;
			std::vector<uint> texcoordsIndices;
			size_t dimensions;
			size_t elementCount;
			size_t vertexCount;
			uint elementSize;
			BBox bbox;
	};

} // ponos namespace

#endif // PONOS_STRUCTURES_RAW_MESH_H

