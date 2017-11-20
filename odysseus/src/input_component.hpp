#ifndef ODYSSEUS_INPUT_COMPONENT_H
#define ODYSSEUS_INPUT_COMPONENT_H

#include "component_interface.h"

namespace odysseus {

	class InputComponent : public ComponentInterface {
		public:
			virtual ~InputComponent() {}
			virtual void update(GameObject& obj) = 0;
	};

} // odysseus namespace

#endif // ODYSSEUS_INPUT_COMPONENT_H
