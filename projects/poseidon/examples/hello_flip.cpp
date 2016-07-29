#include <aergia.h>
#include <poseidon.h>
#include <ponos.h>
// #include <windows.h>

#include "flip_drawer.h"

int w = 7, h = 7;
aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
aergia::Camera2D camera;
poseidon::FLIP flip;
FLIPDrawer fd(&flip);

void render(){
  gd.clearScreen(1.f, 1.f, 1.f, 0.f);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  camera.look();
  fd.drawParticles();
  fd.drawMACGrid();
  fd.drawGridVelocities(flip.v, 1);
  fd.drawGridVelocities(flip.u, 0);
  fd.drawCells();

  // static int k = 1;
 // if(k++ > 50)
  //  return;
    std::cout << "STEP\n";
  flip.step();
    std::cout << "STEP END\n";
    //Sleep(100);
}

int main() {
  // WIN32CONSOLE();
  // init FLIP
  flip.set(w, h, ponos::vec2(0.f,0.f), 0.1f);
  flip.gravity = ponos::vec2(0.f, -9.8f);
  flip.dt = 0.001f;
  flip.rho = 1.f;
  for(int i = 0; i < w; i++)
    flip.isSolid(i, 0) = flip.isSolid(i, h - 1) = 1;
  for(int i = 0; i < h; i++)
    flip.isSolid(0, i) = flip.isSolid(w - 1, i) = 1;

  for (int i = 2; i < 5; ++i) {
    for (int j = 2; j < 5; ++j) {
      flip.fillCell(i, j);
    }
  }
  // set camera
  camera.setPos(ponos::vec2(w * flip.dx / 2.f, h * flip.dx / 2.f));
  camera.setZoom(w * flip.dx / 1.5f);
  camera.resize(800,800);
  // init window
  aergia::createGraphicsDisplay(800, 800, "FLIP - 2D");
  gd.registerRenderFunc(render);
  gd.start();

  return 0;
}
