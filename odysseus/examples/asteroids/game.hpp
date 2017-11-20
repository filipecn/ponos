#include <aergia.h>
#include <odysseus.h>
#include <hercules.h>

using namespace odysseus;

#include "player.h"

class AsteroidsGame : public Game {
	public:
		AsteroidsGame() {}
		~AsteroidsGame();

	private:
		virtual void init() override;
		virtual void update() override;
		virtual void render() override;
		virtual void processInput() override;

		Player* ship;
		ponos::ObjectPool<Player, 1000> asteroids;
		hercules::_2d::World world;

		aergia::Camera2D camera;
};

AsteroidsGame::~AsteroidsGame() {
	delete ship;
}

void AsteroidsGame::init() {
	// create ship
	ship = createShip(world);
	// set camera
	camera.setPos(ponos::vec2(0.f, 0.f));
	camera.setZoom(3.f);
	camera.resize(800,800);
	for(int i = 0; i < 50; i++) {
		Player* newAsteroid = asteroids.create();
		createAsteroid(newAsteroid, world);
	}
}

void AsteroidsGame::update() {}

void AsteroidsGame::render() {
	aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
	gd.clearScreen(1,1,1,1);
	glEnable (GL_BLEND);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	camera.look();
	if(ship)
		ship->render();
	for(ponos::ObjectPool<Player, 1000>::iterator it(asteroids); it.next(); ++it) {
		(*it)->render();
	}
}

void AsteroidsGame::processInput() {
	ship->processInput();
	aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
	if(gd.keyState(GLFW_KEY_SPACE) == GLFW_PRESS) {
	//	bulletManager->shoot(ship->transform, world);
	}
}

