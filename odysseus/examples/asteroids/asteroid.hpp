#include <odysseus.h>

class Asteroid : public odysseus::GameObject {
	public:
		Asteroid(
				std::shared_ptr<GraphicsComponent> graphics,
				std::shared_ptr<PhysicsComponent> physics) {
			graphics_ = graphics;
			physics_ = physics;
		}

		virtual void render() override {
			if(graphics_)
				graphics_->render(this);
		}
		virtual void update() override {}
		virtual void processInput() override {}

};

class AsteroidGraphics : public GraphicsComponent {
	public:
		AsteroidGraphics() {
			vertices.emplace_back(0.0f, 0.0f);
			vertices.emplace_back(1.0f, 0.0f);
			vertices.emplace_back(1.0f, 1.0f);
			vertices.emplace_back(0.0f, 1.0f);
		}
		virtual void render(GameObject* Asteroid) override {
			ponos::Transform2D t = Asteroid->transform * transform;
			glColor3f(1,0,0);
			glBegin(GL_LINE_LOOP);
			aergia::glVertex(t(vertices[0]));
			aergia::glVertex(t(vertices[1]));
			aergia::glVertex(t(vertices[2]));
			aergia::glVertex(t(vertices[3]));
			glEnd();
		}
};

class AsteroidPhysics : public PhysicsComponent {
	public:
		AsteroidPhysics() {}
		virtual void update(std::shared_ptr<GameObject> asteroid, hercules::World &world) override {
		}
};
