#ifndef ODYSSEUS_GAME_OBJECT_H
#define ODYSSEUS_GAME_OBJECT_H

#include "graphics_component.hpp"
#include "input_component.hpp"
#include "physics_component.hpp"

#include <memory>

#include <hercules.h>
#include <ponos.h>

namespace odysseus {

	class GameObject {
		public:
			GameObject() {}
			GameObject(GraphicsComponent *graphics,
					InputComponent *input,
					PhysicsComponent *physics) {
				graphics_.reset(graphics);
				input_.reset(input);
				physics_.reset(physics);
			}
			virtual ~GameObject() {}

			void update(hercules::_2d::World  &world, GraphicsManager &graphics) {
				input_->update(*this);
				physics_->update(*this, world);
				graphics_->update(*this, graphics);
			}

			GraphicsComponent* getGraphicsComponent() { return graphics_.get(); }
			PhysicsComponent* getPhysicsComponent() { return physics_.get(); }
			InputComponent* getInputComponent() { return input_.get(); }

			ponos::Transform2D transform;

		protected:
			std::shared_ptr<GraphicsComponent> graphics_;
			std::shared_ptr<InputComponent> input_;
			std::shared_ptr<PhysicsComponent> physics_;
	};

} // odysseus namespace

#endif // ODYSSEUS_GAME_OBJECT_H
