#include <aergia.h>
#include <ponos.h>
#include <iostream>
#include <vector>

aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
aergia::CartesianGrid grid;
aergia::Camera camera;
ponos::Transform toWorld;
aergia::TrackballInterface trackball;

void button(int button, int action) {
	if(action == GLFW_RELEASE)
		trackball.buttonRelease(camera, button);
	else
		trackball.buttonPress(camera, button);
}

void mouse(double x, double y) {
	trackball.mouseMove(camera);
}

void render() {
  gd.clearScreen(1.f, 1.f, 1.f, 0.f);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  camera.look();
	grid.draw();
	glPointSize(5);
	glBegin(GL_POINTS);
	glColor3f(0,0,0);
	aergia::glVertex(ponos::Point3(0.5,0.5,0));
	glEnd();
}

int main() {
	// set grid
	grid.setDimension(0, -5, 5);
	grid.setDimension(1, -5, 5);
	grid.setDimension(2, -5, 5);
	// set camera
  camera.resize(800,800);
  toWorld = ponos::Transform(ponos::inverse(camera.getTransform()));
  // init window
  aergia::createGraphicsDisplay(800, 800, "HELLO 3D");
  gd.registerRenderFunc(render);
  gd.registerButtonFunc(button);
  gd.registerMouseFunc(mouse);
  gd.start();
	return 0;
}
