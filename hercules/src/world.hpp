#pragma once

#include <ponos.h>

#include <vector>

namespace hercules {

	class World {
		public:
			World() {}

			Body* createBody();

		private:
			ponos::ObjectPool<Body, 20> bodies;
	};

	Body* World::createBody() {
		return bodies.create();
	}

} // hercules namespace
