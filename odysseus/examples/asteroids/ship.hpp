#include <odysseus.h>

using namespace odysseus;

class Ship : public GameObject {
	public:
		Ship(
				std::shared_ptr<GraphicsComponent> graphics,
				std::shared_ptr<InputComponent> input,
				std::shared_ptr<PhysicsComponent> physics)
			: GameObject(graphics, input, physics) {
			}

		virtual void render() override {
			if(graphics_)
				graphics_->render(this);
		}
		virtual void processInput() override {
			if(input_)
				input_->processInput(this);
		}
		virtual void update() override {}
};

class ShipInput : public InputComponent {
	public:
		virtual void processInput(GameObject* ship) override {
			aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
			if(gd.keyState(GLFW_KEY_LEFT) == GLFW_PRESS)
				ship->transform = ship->transform * ponos::rotate(0.001f);
			if(gd.keyState(GLFW_KEY_RIGHT) == GLFW_PRESS)
				ship->transform = ship->transform * ponos::rotate(-0.001f);
		}
};

class ShipGraphics : public GraphicsComponent {
	public:
		ShipGraphics() {
			vertices.emplace_back(0.0f, 0.0f);
			vertices.emplace_back(1.0f, 0.0f);
			vertices.emplace_back(0.5f, 1.0f);
			transform.translate(ponos::vec2(-0.5f, -0.5f));
		}
		virtual void render(GameObject* ship) override {
			ponos::Transform2D t = ship->transform * transform;
			glColor3f(0,0,0);
			glBegin(GL_LINE_LOOP);
				aergia::glVertex(t(vertices[0]));
				aergia::glVertex(t(vertices[1]));
				aergia::glVertex(t(vertices[2]));
			glEnd();
		}
};

class ShipPhysics : public PhysicsComponent {
	public:
		ShipPhysics(hercules::World &world) {
			// body.reset(world.createBody());
			hercules::Fixture *fixture = new hercules::Fixture();
			hercules::Polygon *polygon = new hercules::Polygon();
			polygon->vertices.emplace_back(0.f, 0.f);
			polygon->vertices.emplace_back(1.f, 0.f);
			polygon->vertices.emplace_back(0.5f, 1.f);
			fixture->shape.reset(polygon);
			body->fixture.reset(fixture);
		}
		virtual void update(std::shared_ptr<GameObject> ship, hercules::World &world) override {
		}
};
