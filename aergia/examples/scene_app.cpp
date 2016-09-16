#include <aergia.h>
#include <ponos.h>
#include <iostream>
#include <vector>

int WIDTH = 800, HEIGHT = 800;

aergia::SceneApp<> app(WIDTH, HEIGHT, "Scene App");

int main() {
	app.scene.add(new aergia::CartesianGrid(5, 5, 5));
	app.run();
	return 0;
}
