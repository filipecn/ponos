#ifndef PONOS_SPATIAL_SPATIAL_GRID_H
#define PONOS_SPATIAL_SPATIAL_GRID_H

#include "common/memory.h"
#include "geometry/numeric.h"
#include "geometry/interval.h"
#include "geometry/ray.h"
#include "spatial/spatial_structure_interface.h"

#include <functional>
#include <stack>

namespace hercules {

	namespace cds {

		/* Spatial 2D Grid
		 * Defines a 2D grid space partitioning scheme for objects organization.
		 *
		 * **Note** The order of insertion of objects is not kept.
		 *
		 * **ActiveObjectType must have these public methods (or implement ActiveObject's Interface):
		 *
		 * **bool** isActive() -> **false** if object can be deleted
		 * **BBox2D** getWBBox() -> bbox in world coordinates
		 */
		template<typename AABBObjectType>
				 class AABBGrid2D : public ponos::CObjectPool<AABBObjectType> {
					 public:
						 virtual ~AABBGrid2D() {}
				 };

	} // cds namespace

} // ponos namespace

#endif // PONOS_SPATIAL_SPATIAL_GRID_H

