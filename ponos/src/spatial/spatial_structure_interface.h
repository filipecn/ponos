#ifndef PONOS_SPATIAL_STRUCTURE_INTERFACE_H
#define PONOS_SPATIAL_STRUCTURE_INTERFACE_H

#include <functional>

namespace ponos {

	/* interface
	 * Defines an interface for spatial structures. Acceleration schemes for fast spatial
	 * queries and geometric objects arrangement.
	 */
	template<typename ObjectType>
		class SpatialStructureInterface {
			public:
				SpatialStructureInterface() {}
				virtual ~SpatialStructureInterface() {}
				/* add
				 * @o **[in]** object
				 * Add **o** to the structure
				 */
				virtual void add(const ObjectType& o) = 0;
				/* iterate
				 * @f **[in]** function called for each object
				 * Iterates thourgh all objects in the structure
				 */
				virtual void iterate(std::function<void(const ObjectType& o)> f) = 0;
		};

} // ponos namespace

#endif // PONOS_SPATIAL_STRUCTURE_INTERFACE_H

