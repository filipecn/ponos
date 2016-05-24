#include <aergia.h>
#include <poseidon.h>
#include <ponos.h>

#include "flip_drawer.h"

aergia::Camera2D camera;
poseidon::FLIP flip;
FLIPDrawer fd(&flip);

void render(){
	glClearColor(1,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  camera.look();
  glPointSize(5.0);
  glColor3f(1,1,1);
  glBegin(GL_POINTS);
    glVertex2f(0.0,0.0);
  glEnd();
  fd.drawGrid();
}

int main() {
  WIN32CONSOLE();
  // set camera
  camera.setZoom(15.f);
  camera.resize(800,800);
  // init FLIP
  flip.set(10, 10, ponos::vec2(0.f,0.f), 1.f);
  // init window
  aergia::GraphicsDisplay& gd = aergia::createGraphicsDisplay(800, 800, "Hello Aergia");
  gd.registerRenderFunc(render);
  gd.start();

  return 0;
}
