#include <aergia.h>

using namespace aergia;

int WIDTH = 800, HEIGHT = 800;
GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
App app(WIDTH, HEIGHT);

void button(int button, int action) {
	app.processInput();
}

void mouse(double x, double y) {
	app.processInput();
}

void render() {
	app.render();
}

int main() {
  // init window
  createGraphicsDisplay(WIDTH, HEIGHT, "HELLO 3D");
  gd.registerRenderFunc(render);
  gd.registerButtonFunc(button);
  gd.registerMouseFunc(mouse);
  gd.start();
	return 0;
}
