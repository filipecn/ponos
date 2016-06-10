#include <aergia.h>
#include <poseidon.h>
#include <ponos.h>

aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
aergia::Camera2D camera;
Transform toWorld;
poseidon::ParticleGrid particleGrid;
int w = 5, h = 5;
float dx = 0.1;
enum CellType {FLUID = 1, AIR, SOLID, CELLTYPES};
ponos::ZGrid<char> cell;
int selected = -1;

ponos::Point2 newPosition() {
  poseidon::Particle2D pa = particleGrid.getParticle(0);
  ponos::Point2 e = particleGrid.getParticle(1).p;
  int curCell = cell.dSample(e.x, e.y, -1);
  if(curCell < 0 || curCell == CellType::SOLID) {
    while(ponos::distance(pa.p, e) > 1e-5 && (curCell < 0 || curCell == CellType::SOLID)) {
      Vector2 m = 0.5f * Vector2(pa.p.x, pa.p.y) + 0.5f * Vector2(e.x, e.y);
      curCell = cell.dSample(m.x, m.y, -1);
      if(curCell < 0 || curCell == CellType::SOLID)
      e = ponos::Point2(m.x, m.y);
      else pa.p = ponos::Point2(m.x, m.y);
      curCell = cell.dSample(e.x, e.y, -1);
    }
  }
  return e;
}

template<class T>
void drawGrid(const ponos::ZGrid<T>& grid) {
  float hdx = dx / 2.f;
  ponos::vec2 offset = grid.toWorld.getTranslate();
  glBegin(GL_LINES);
  for (int i = 0; i <= grid.width; ++i) {
    glVertex2f(static_cast<float>(i) * dx + offset.x - hdx, offset.y - hdx);
    glVertex2f(static_cast<float>(i) * dx + offset.x - hdx,
    offset.y + dx * grid.height - hdx);
  }
  for (int i = 0; i <= grid.height; ++i) {
    glVertex2f(offset.x - hdx, static_cast<float>(i) * dx + offset.y - hdx);
    glVertex2f(offset.x + dx * grid.width - hdx,
    static_cast<float>(i) * dx + offset.y - hdx);
  }
  glEnd();
}

void drawCellGrid() {
  glColor4f(0.5, 0.0, 0.8, 0.2);
  drawGrid<char>(cell);
  for (int i = 0; i < cell.width; ++i)
  for (int j = 0; j < cell.height; ++j) {
    glColor4f(1,1,1,0);
    if(cell(i, j) == FLUID)
      glColor4f(0.1, 0.0, 0.8, 0.2);
    if(cell(i, j) == SOLID)
      glColor4f(0.0, 0.0, 0.0, 0.2);
    ponos::Point2 wp = cell.toWorld(ponos::Point2(i, j));
    glBegin(GL_QUADS);
      aergia::glVertex(wp + ponos::vec2(-0.5 * dx, -0.5 * dx));
      aergia::glVertex(wp + ponos::vec2( 0.5 * dx, -0.5 * dx));
      aergia::glVertex(wp + ponos::vec2( 0.5 * dx,  0.5 * dx));
      aergia::glVertex(wp + ponos::vec2(-0.5 * dx,  0.5 * dx));
    glEnd();
    glColor3f(0,0,0);
    glPointSize(4.0);
    glBegin(GL_POINTS);
      aergia::glVertex(wp);
    glEnd();
  }
}

void drawParticle(poseidon::Particle2D particle) {
  aergia::Circle c;
  c.r = 0.1 * dx;
  c.p.x = particle.p.x;
  c.p.y = particle.p.y;
  c.draw();
}

void render(){
  gd.clearScreen(1.f, 1.f, 1.f, 0.f);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  camera.look();
  drawCellGrid();
  glColor4f(1,0,0,0.8);
  drawParticle(particleGrid.getParticle(0));
  glColor4f(0,0,1,0.8);
  drawParticle(particleGrid.getParticle(1));
  glBegin(GL_LINES);
    aergia::glVertex(particleGrid.getParticle(0).p);
    aergia::glVertex(particleGrid.getParticle(1).p);
  glEnd();
  glPointSize(5);
  glColor4f(0,1,0.4,0.8);
  glBegin(GL_POINTS);
    aergia::glVertex(newPosition());
  glEnd();
}

void updateCells() {
  for(int i = 0; i < w; i++)
  for(int j = 0; j < h; j++) {
    if(i == 0 || i == w - 1 || j == 0 || j == h - 1)
      cell(i, j) = SOLID;
    else if(particleGrid.grid(i, j).numberOfParticles())
      cell(i, j) = FLUID;
    else cell(i, j) = AIR;
  }
}

ponos::Point2 worldMousePoint() {
    ponos::Point2 mp = gd.getMouseNPos();
    return toWorld(ponos::Point3(mp.x, mp.y, 1.f)).xy();
}

void button(int button, int action) {
  if(action == GLFW_PRESS) {
    ponos::Point2 wp = worldMousePoint();
    for (int i = 0; i < 2; ++i) {
      if(ponos::distance(wp, particleGrid.getParticle(i).p) < dx * 0.1f)
        selected = i;
    }
  }
  else selected = -1;
}

void mouse(double x, double y) {
  if (selected >= 0) {
    particleGrid.setPos(selected, worldMousePoint());
    updateCells();
  }
}

int main() {
  WIN32CONSOLE();
  particleGrid.set(w, h, ponos::vec2(0.f, 0.f), ponos::vec2(dx, dx));
  cell.set(w, h, ponos::vec2(0.f, 0.f), ponos::vec2(dx, dx));
  cell.init();
  for(int i = 0; i < w; i++)
  for(int j = 0; j < h; j++)
    if(i == 0 || i == w - 1 || j == 0 || j == h - 1)
      cell(i, j) = SOLID;
    else cell(i, j) = AIR;

  particleGrid.addParticle(1, 1, poseidon::Particle2D(cell.toWorld(ponos::Point2(1, 1)), ponos::Vector2()));
  particleGrid.addParticle(2, 2, poseidon::Particle2D(cell.toWorld(ponos::Point2(2, 2)), ponos::Vector2()));

  // set camera
  camera.setPos(ponos::vec2(w * dx / 2.f, h * dx / 2.f));
  camera.setZoom(w * dx / 1.5f);
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
