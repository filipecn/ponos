#pragma once

#include <ponos.h>

namespace odysseus {

	class GameObject;

	class GraphicsComponent {
		public:
			virtual ~GraphicsComponent() {}
			virtual void render(GameObject* obj) = 0;

			ponos::Transform2D transform;
			std::vector<ponos::Point2> vertices;
	};

} // odysseus namespace
