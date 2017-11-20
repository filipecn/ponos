#ifndef ODYSSEUS_PHYSICS_COMPONENT_H
#define ODYSSEUS_PHYSICS_COMPONENT_H

#include "component_interface.h"

#include <hercules.h>

#include <memory>

namespace odysseus {

	class PhysicsComponent : public ComponentInterface {
		public:
			PhysicsComponent(hercules::_2d::RigidBodyPtr b)
				: body(b) {}
			virtual ~PhysicsComponent() {}
			virtual void update(GameObject &obj, hercules::_2d::World &world) = 0;

		protected:
			hercules::_2d::RigidBodyPtr body;
	};

} // odysseus namespace

#endif // ODYSSEUS_PHYSICS_COMPONENT_H
