#include <aergia.h>
#include <poseidon.h>
#include <ponos.h>
#include <stdlib.h>
#include <flip_drawer.h>

poseidon::FLIP flip;
FLIPDrawer fd(&flip);
int w = 5, h = 5;
aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
aergia::Camera2D camera;
Transform toWorld;
int selected = -1;

void render(){
  gd.clearScreen(1.f, 1.f, 1.f, 0.f);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  camera.look();
  flip.enforceBoundary();
  flip.classifyCells();
  // flip.scatter(flip.u, 0);
  flip.scatter(flip.v, 1);
  fd.drawParticles();
  fd.drawMACGrid();
  fd.drawGridVelocities(flip.v, 1);
  fd.drawGridVelocities(flip.u, 0);
  fd.drawCells();
}

ponos::Point2 worldMousePoint() {
    ponos::Point2 mp = gd.getMouseNPos();
    return toWorld(ponos::Point3(mp.x, mp.y, 1.f)).xy();
}

void button(int button, int action) {
  if(action == GLFW_PRESS) {
    ponos::Point2 wp = worldMousePoint();
    const std::vector<poseidon::Particle2D>& particles = flip.particleGrid.getParticles();
    for (int i = 0; i < particles.size(); ++i) {
      if(ponos::distance(wp, particles[i].p) < flip.dx * 0.1f)
        selected = i;
    }
  }
  else selected = -1;
}

void mouse(double x, double y) {
  if (selected >= 0) {
    flip.particleGrid.setPos(selected, worldMousePoint());
  }
}

int main() {
  WIN32CONSOLE();
  // init FLIP
  flip.set(w, h, ponos::vec2(0.f,0.f), 0.1f);
  flip.gravity = ponos::vec2(0.f, -9.8f);
  flip.dt = 0.005f;
  flip.rho = 1.f;
  for(int i = 0; i < w; i++)
  flip.isSolid(i, 0) = flip.isSolid(i, h - 1) = 1;
  for(int i = 0; i < h; i++)
  flip.isSolid(0, i) = flip.isSolid(w - 1, i) = 1;

  srand(123);
  for (int i = 0; i < w; ++i)
  {
    for (int j = 0; j < h; ++j)
    {
      flip.u(i, j).v = rand() % 10;
      flip.v(i, j).v = rand() % 15;
    }
  }

  flip.particleGrid.addParticle(1, 1, poseidon::Particle2D(flip.cell.toWorld(ponos::Point2(1, 1)), ponos::Vector2()));
  flip.particleGrid.addParticle(2, 2, poseidon::Particle2D(flip.cell.toWorld(ponos::Point2(2, 2)), ponos::Vector2()));

  // set camera
  camera.setPos(ponos::vec2(w * flip.dx / 2.f, h * flip.dx / 2.f));
  camera.setZoom(w * flip.dx / 1.5f);
  camera.resize(800,800);
  toWorld = ponos::Transform(ponos::inverse(camera.getTransform()));
  // init window
  aergia::createGraphicsDisplay(800, 800, "FLIP - 2D");
  gd.registerRenderFunc(render);
  gd.registerButtonFunc(button);
  gd.registerMouseFunc(mouse);
  gd.start();

  return 0;
}
