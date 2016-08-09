#include <aergia.h>
#include <odysseus.h>

using namespace odysseus;

#include "asteroid.hpp"
#include "ship.hpp"
#include "bullet_pool.hpp"

class AsteroidsGame : public Game {
	public:
		AsteroidsGame() {}

	private:
		virtual void init() override;
		virtual void update() override;
		virtual void render() override;
		virtual void processInput() override;

		std::unique_ptr<Ship> ship;
		std::unique_ptr<BulletManager> bulletManager;
		std::vector<Asteroid> asteroids;

		aergia::Camera2D camera;
		hercules::World world;
};

void AsteroidsGame::init() {
	// create ship
	ship.reset(new Ship(
				std::shared_ptr<GraphicsComponent>(new ShipGraphics()),
				std::shared_ptr<InputComponent>(new ShipInput()),
				std::shared_ptr<PhysicsComponent>(new ShipPhysics(world))));
	bulletManager.reset(new BulletManager(20));
	std::shared_ptr<GraphicsComponent> asteroidGraphics(new AsteroidGraphics);
	std::shared_ptr<PhysicsComponent> asteroidPhysics(new AsteroidPhysics);
	asteroids.emplace_back(asteroidGraphics, asteroidPhysics);
	asteroids.emplace_back(asteroidGraphics, asteroidPhysics);
	// set camera
	camera.setPos(ponos::vec2(0.f, 0.f));
	camera.setZoom(3.f);
	camera.resize(800,800);
}

void AsteroidsGame::update() {}

void AsteroidsGame::render() {
	aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
	gd.clearScreen(1,1,1,1);
	glEnable (GL_BLEND);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	camera.look();
	ship->render();
	for(Asteroid a : asteroids) {
		a.render();
	}
}

void AsteroidsGame::processInput() {
	ship->processInput();
	aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
	if(gd.keyState(GLFW_KEY_SPACE) == GLFW_PRESS) {
		bulletManager->shoot(ship->transform, world);
	}
}

