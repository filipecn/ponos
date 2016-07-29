#pragma once

#include "graphics_component.hpp"
#include "input_component.hpp"
#include "physics_component.hpp"

#include <memory>

#include <ponos.h>

namespace odysseus {

	class GameObject {
		public:
			GameObject() {}
			GameObject(
					std::shared_ptr<GraphicsComponent> graphics,
					std::shared_ptr<InputComponent> input,
					std::shared_ptr<PhysicsComponent> physics) :
				graphics_(graphics),
				input_(input),
				physics_(physics) {}
			virtual ~GameObject() {}

			virtual void render() = 0;
			virtual void processInput() = 0;
			virtual void update() = 0;

			ponos::Transform2D transform;

		protected:
			std::shared_ptr<GraphicsComponent> graphics_;
			std::shared_ptr<InputComponent> input_;
			std::shared_ptr<PhysicsComponent> physics_;
	};

} // odysseus namespace
