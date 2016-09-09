#include <aergia.h>
#include <ponos.h>
#include <iostream>
#include <vector>
#include <memory>

int WIDTH = 800, HEIGHT = 800;

aergia::App app(WIDTH, HEIGHT, "FLIP example");
std::shared_ptr<aergia::CartesianGrid> grid;
aergia::Scene<> scene;

int main() {
	app.viewports[0].camera.reset(new aergia::Camera());
	app.viewports[0].camera->setPos(ponos::Point3(40.f, 10.f, 10.f));
	// set grid
	scene.add(new aergia::CartesianGrid(5, 5, 5));
	// callbacks
	app.viewports[0].renderCallback = []() { scene.render(); };
	app.run();
	return 0;
}
