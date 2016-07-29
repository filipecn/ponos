#pragma once

#include <hercules.h>

#include <memory>

namespace odysseus {

	class PhysicsComponent {
		public:
			virtual ~PhysicsComponent() {}
			virtual void update(std::shared_ptr<GameObject> obj, hercules::World &world) = 0;

		protected:
			std::shared_ptr<hercules::Body> body;
	};

} // odysseus namespace
