#ifndef HERCULES_2D_WORLD_H
#define HERCULES_2D_WORLD_H

#include "cds/cds.h"
#include "world_interface.h"
#include "2d/rigid_body.h"

#include <ponos.h>

#include <vector>

namespace hercules {

	namespace _2d {

		typedef ponos::IndexPointer<RigidBody> RigidBodyPtr;

		class World : public WorldInterface {
			public:
				World() {}

				template<typename... Args>
					RigidBodyPtr create(Args&&... args) {
						return objects.create(std::forward<Args>(args)...);
					}
				void destroy(RigidBodyPtr body) {
					objects.destroy(body);
				}
			private:
				//cds::AABBGrid2D<RigidBody> objects;
				cds::AABBSweep<RigidBody> objects;
		};

	} // 2d namespace

} // hercules namespace

#endif
