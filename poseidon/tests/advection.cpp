#include <aergia.h>
#include <poseidon.h>
#include <ponos.h>

#include <flip_drawer.h>

aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
aergia::Camera2D camera;
Transform toWorld;
poseidon::FLIP flip;
FLIPDrawer fd(&flip);
int w = 5, h = 5;
int selected = -1;

void render(){
  gd.clearScreen(1.f, 1.f, 1.f, 0.f);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  camera.look();
  flip.classifyCells();
  fd.drawMACGrid();
  fd.drawGridVelocities(flip.v, 1);
  fd.drawGridVelocities(flip.u, 0);
  fd.drawCells();
  glColor4f(0.f,0.5f,0.2f,0.5f);
  fd.drawParticle(flip.particleGrid.getParticle(0));
  fd.drawParticle(flip.particleGrid.getParticle(1));
  glColor4f(0.f,0.5f,0.2f,0.2f);
  fd.drawParticle(flip.particleGrid.getParticle(2));
}

ponos::Point2 worldMousePoint() {
    ponos::Point2 mp = gd.getMouseNPos();
    return toWorld(ponos::Point3(mp.x, mp.y, 1.f)).xy();
}

void button(int button, int action) {
  if(action == GLFW_PRESS) {
    ponos::Point2 wp = worldMousePoint();
    for (int i = 0; i < 2; ++i) {
      if(ponos::distance(wp, flip.particleGrid.getParticle(i).p) < flip.dx * 0.1f)
        selected = i;
    }
  }
  else selected = -1;
}

void mouse(double x, double y) {
  if (selected >= 0) {
    flip.particleGrid.setPos(selected, worldMousePoint());
    ponos::vec2 d = flip.particleGrid.getParticle(1).p - flip.particleGrid.getParticle(0).p;
    d /= flip.dt;
    flip.particleGrid.getParticleReference(0).v = d;
    flip.particleGrid.setPos(2, flip.newPosition(flip.particleGrid.getParticle(0)));
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

  flip.particleGrid.addParticle(1, 1, poseidon::Particle2D(flip.cell.toWorld(ponos::Point2(1, 1)), ponos::Vector2()));
  flip.particleGrid.addParticle(2, 2, poseidon::Particle2D(flip.cell.toWorld(ponos::Point2(2, 2)), ponos::Vector2()));
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
