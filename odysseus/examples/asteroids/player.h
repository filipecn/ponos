#include <odysseus.h>

#include <stdlib.h>
#include <time.h>

using namespace odysseus;

class Player : public odysseus::GameObject {
	public:
		Player() {}
		Player(GraphicsComponent *graphics,
				InputComponent *input,
				PhysicsComponent *physics) :
			GameObject(graphics, input, physics) {
			}
		void render() {
			graphics_->update(*this, odysseus::GraphicsManager::instance());
		}
		void processInput() {
			if(input_)
				input_->update(*this);
		}
		void setGraphics(GraphicsComponent *g) { graphics_.reset(g); }
		void setPhysics(PhysicsComponent *p) { physics_.reset(p); }
};

class ShipInput : public odysseus::InputComponent {
	public:
		virtual void update(GameObject &ship) override {
			aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
			if(gd.keyState(GLFW_KEY_LEFT) == GLFW_PRESS)
				ship.transform = ship.transform * ponos::rotate(1.f);
			if(gd.keyState(GLFW_KEY_RIGHT) == GLFW_PRESS)
				ship.transform = ship.transform * ponos::rotate(-1.f);
		}
};

class PlayerGraphics : public odysseus::GraphicsComponent {
	public:
		PlayerGraphics(std::vector<ponos::Point2> vertices, aergia::Shader s, float _r, float _g, float _b) {
			shape.vertices = vertices;
			shader = s;
			r = _r;
			g = _g;
			b = _b;
		}
		virtual void update(GameObject &Player, GraphicsManager &graphics) override {
			shader.begin();
			shader.setUniform("color", ponos::vec4(r, g, b, 1.f));
			glBegin(GL_LINE_LOOP);
			for(ponos::Point2& p : shape.vertices)
				aergia::glVertex(Player.transform(p));
			glEnd();
			shader.end();
		}
	private:
		ponos::Polygon shape;
		aergia::Shader shader;
		float r, g, b;
};

class PlayerPhysics : public odysseus::PhysicsComponent {
	public:
		PlayerPhysics(hercules::_2d::RigidBodyPtr b)
		: odysseus::PhysicsComponent(b) {}
		virtual void update(GameObject &Player, hercules::_2d::World &world) override {
		}
};

void createAsteroid(Player* asteroid, hercules::_2d::World &world) {
	static int sid = aergia::ShaderManager::instance().loadFromTexts(
			nullptr,
			nullptr,
			"#version 440 core\n"
			 "in vec4 color;"
			 "out vec4 outColor;"
			 "void main() {"
    	"	outColor = color;"
			 "}"
			);
	static aergia::Shader s(sid);
	std::vector<ponos::Point2> vertices;
	vertices.emplace_back(0.0f, 0.0f);
	vertices.emplace_back(1.0f, 0.0f);
	vertices.emplace_back(1.0f, 1.0f);
	vertices.emplace_back(0.0f, 1.0f);
	asteroid->setGraphics(new PlayerGraphics(vertices, s, 0, 0, 0));
	asteroid->setPhysics(new PlayerPhysics(world.create(nullptr, new ponos::Polygon(vertices))));
	//srand (time(NULL));
	ponos::vec2 t(
			(1.f * rand()) / (1.f * RAND_MAX),
			(1.f * rand()) / (1.f * RAND_MAX));
	asteroid->transform.translate(t);
}

Player* createShip(hercules::_2d::World &world) {
	static int sid = aergia::ShaderManager::instance().loadFromTexts(
			nullptr,
			nullptr,
			"#version 440 core\n"
			 "uniform vec4 color;"
			 "out vec4 outColor;"
			 "void main() {"
    	"	outColor = vec4(color.x, 0, 0, 1);"
			 "}"
			);
	static aergia::Shader s(sid);
	std::vector<ponos::Point2> vertices;
	vertices.emplace_back(0.0f, 0.0f);
	vertices.emplace_back(0.5f, 0.0f);
	vertices.emplace_back(0.5f, 1.0f);
	vertices.emplace_back(0.0f, 1.0f);
	Player* ship = new Player(
			new PlayerGraphics(vertices, s, 1, 1, 0),
			new ShipInput(),
			new PlayerPhysics(world.create(new hercules::_2d::Fixture(), new ponos::Polygon(vertices))));
	return ship;
}
