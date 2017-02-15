#include <aergia.h>
#include <ponos.h>
#include <iostream>
#include <vector>

int WIDTH = 1600, HEIGHT = 800;

aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
aergia::ViewportDisplay cameraview(0, 0, WIDTH / 2, HEIGHT);
aergia::ViewportDisplay worldview(WIDTH / 2, 0, WIDTH / 2, HEIGHT);
aergia::CartesianGrid grid;
aergia::TrackballInterface trackball;

void button(int button, int action) {
	if(action == GLFW_RELEASE)
		trackball.buttonRelease(*cameraview.camera.get(), button, cameraview.getMouseNPos());
	else
		trackball.buttonPress(*cameraview.camera.get(), button, cameraview.getMouseNPos());
}

void mouse(double x, double y) {
	trackball.mouseMove(*cameraview.camera.get(), cameraview.getMouseNPos());
//	cameraview.camera->setPos(trackball.tb.transform(cameraview.camera->getPos()));
//	cameraview.camera->setTarget(trackball.tb.transform(cameraview.camera->getTarget()));
//	trackball.tb.center = cameraview.camera->getTarget();
	grid.transform = trackball.tb.transform * grid.transform;
}

void renderCameraView() {
	grid.draw();
	trackball.draw();
}

void renderWorldView() {
	grid.draw();
	aergia::CameraModel::drawCamera(*static_cast<aergia::Camera*>(cameraview.camera.get()));
	trackball.draw();
}

void render() {
	cameraview.render();
	worldview.render();
}

int main() {
	// set camera view
	cameraview.camera.reset(new aergia::Camera());
	cameraview.camera->resize(WIDTH / 2, HEIGHT);
	static_cast<aergia::Camera*>(cameraview.camera.get())->setFar(40.f);
  static_cast<aergia::Camera*>(cameraview.camera.get())->setNear(2.f);
	cameraview.renderCallback = renderCameraView;
	// set world view
	worldview.camera.reset(new aergia::Camera());
	worldview.camera->resize(WIDTH / 2, HEIGHT);
  static_cast<aergia::Camera*>(worldview.camera.get())->setPos(ponos::Point3(40.f, 10.f, 10.f));
	worldview.renderCallback = renderWorldView;
	// set grid
	grid.setDimension(0, -5, 5);
	grid.setDimension(1, -5, 5);
	grid.setDimension(2, -5, 5);
  // init window
  aergia::createGraphicsDisplay(WIDTH, HEIGHT, "HELLO 3D");
  gd.registerRenderFunc(render);
  gd.registerButtonFunc(button);
  gd.registerMouseFunc(mouse);
  gd.start();
	return 0;
}
