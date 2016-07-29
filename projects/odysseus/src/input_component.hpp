#pragma once

namespace odysseus {

	class GameObject;

	class InputComponent {
		public:
			virtual ~InputComponent() {}
			virtual void processInput(GameObject* obj) {}
	};

} // odysseus namespace
