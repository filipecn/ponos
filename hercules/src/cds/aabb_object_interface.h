#ifndef HERCULES_CDS_AABB_OBJECT_INTERFACE_H
#define HERCULES_CDS_AABB_OBJECT_INTERFACE_H

#include <ponos.h>

namespace hercules {

	namespace cds {

		template<typename BBoxType>
		class AABBObjectInterface {
			public:
				AABBObjectInterface() {}
				virtual ~AABBObjectInterface() {}
				/* get
				 * @return **false** if object is not active (some systems may use this information
				 * to delete objects.
				 */
				virtual bool isActive() { return true; }
				/* get
				 * @return object's bbox in world space.
				 */
				virtual BBoxType getWBBox() = 0;
		};

	} // cds namespace

} // hercules namespace

#endif // HERCULES_CDS_AABB_OBJECT_INTERFACE_H

