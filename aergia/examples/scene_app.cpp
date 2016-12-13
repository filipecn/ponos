#include <aergia.h>
#include <ponos.h>
#include <iostream>
#include <vector>

int WIDTH = 800, HEIGHT = 800;

aergia::SceneApp<> app(WIDTH, HEIGHT, "Scene App");

int main() {
	ponos::RawMesh m;
	aergia::loadOBJ("/home/filipecn/Desktop/dragon.obj", &m);
	m.computeBBox();
	m.splitIndexData();
	m.apply(ponos::scale(10.f, 10.f, 10.f));
	app.init();
	app.scene.add(new aergia::CartesianGrid(5, 5, 5));
	app.scene.add(new aergia::TriangleMesh(&m));
	app.run();
	return 0;
}
