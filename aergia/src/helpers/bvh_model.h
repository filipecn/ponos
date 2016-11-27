#ifndef AERGIA_HELPERS_BVH_MODEL_H
#define AERGIA_HELPERS_BVH_MODEL_H

#include <ponos.h>

#include "helpers/geometry_drawers.h"
#include "scene/bvh.h"
#include "scene/scene_object.h"
#include "utils/open_gl.h"

namespace aergia {

	/* bvh model
	 * Draw BVH nodes
	 */
	class BVHModel : public SceneObject  {
		public:
			BVHModel(const BVH* _bvh)
			: bvh(_bvh) { }
			/* @inherit */
			void draw() const override {
				glColor4f(1, 0, 0, 0.2);
				glLineWidth(1.f);
				for(int i = 0; i < bvh->nodes.size(); i++)
					if(bvh->nodes[i].nElements == 1)
					draw_bbox(bvh->sceneMesh->transform(bvh->nodes[i].bounds));
			}

		private:
			const BVH* bvh;
	};

} // aergia namespace

#endif // AERGIA_HELPERS_CARTESIAN_GRID_H
