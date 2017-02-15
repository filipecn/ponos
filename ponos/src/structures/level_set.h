#ifndef PONOS_STRUCTURES_LEVEL_SET_H
#define PONOS_STRUCTURES_LEVEL_SET_H

#include "structures/c_regular_grid.h"
#include "structures/raw_mesh.h"

#include <memory>

namespace ponos {

	class LevelSet : CRegularGrid<float> {
		public:
			LevelSet(const ivec3& d, const vec3 cellSize = vec3(1.f), const vec3& offset = vec3());
			void setMesh(const RawMesh* mesh);

		private:
			std::shared_ptr<const RawMesh> mesh;
	};

	class LevelSet2D : CRegularGrid2D<float> {
		public:
			LevelSet2D(const ponos::RawMesh* m, const ponos::Transform2D& t);
			LevelSet2D(const ivec2& d, const vec2 cellSize = vec2(1.f), const vec2& offset = vec2());
			void setMesh(const RawMesh* mesh);

			/* merge
			 * @ls **[out]**
			 * @return
			 */
			void merge(const LevelSet2D *ls);
			void copy(const LevelSet2D *ls);


		private:
			std::shared_ptr<const RawMesh> mesh;
	};

} // ponos namespace

#endif // PONOS_STRUCTURES_LEVEL_SET_H

