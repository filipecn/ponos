#ifndef ODYSSEUS_GRAPHICS_COMPONENT_H
#define ODYSSEUS_GRAPHICS_COMPONENT_H

#include "component_interface.h"
#include "graphics_manager.h"

#include <ponos.h>

namespace odysseus {

	class GraphicsComponent : public ComponentInterface {
		public:
			virtual ~GraphicsComponent() {}
			virtual void update(GameObject &obj, GraphicsManager &graphics) = 0;
	};

} // odysseus namespace

#endif // ODYSSEUS_GRAPHICS_COMPONENT_H
