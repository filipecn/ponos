#ifndef ODYSSEUS_COMPONENT_MANAGER_H
#define ODYSSEUS_COMPONENT_MANAGER_H

#include "graphics_component.h"
#include "physics_component.h"
#include "input_component.h"
#include "object_pool.h"

namespace odysseus {

	class ComponentManager {
  	public:
	 		ComponentManager() {}
			virtual ~ComponentManager() {}

		private:
			ObjectPool<PhysicsComponent> physics_components;
			ObjectPool<GraphicsComponent> graphics_components;
			ObjectPool<inputComponent> input_components;
	};

} // odysseus namespace

#endif // ODYSSEUS_COMPONENT_MANAGER_H

