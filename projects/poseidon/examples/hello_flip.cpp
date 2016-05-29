#include <aergia.h>
#include <poseidon.h>
#include <ponos.h>

#include "flip_drawer.h"

aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
aergia::Camera2D camera;
poseidon::FLIP flip;
FLIPDrawer fd(&flip);

void render(){
  gd.clearScreen(1.f, 1.f, 1.f, 0.f);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  camera.look();
  //fd.drawGrid<poseidon::ParticleGrid::ParticleCell>(flip.particleGrid.grid);
  fd.drawParticles();
  fd.drawMACGrid();
}

int main() {
  WIN32CONSOLE();
  // set camera
  camera.setPos(ponos::vec2(2.25, 2.25));
  camera.setZoom(3.f);
  camera.resize(800,800);
  // init FLIP
  flip.set(10, 10, ponos::vec2(0.f,0.f), 0.5f);
  flip.fillCell(5, 5);
  // init window
  aergia::createGraphicsDisplay(800, 800, "FLIP - 2D");
  gd.registerRenderFunc(render);
  gd.start();

  return 0;
}
