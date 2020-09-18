#include <aergia.h>
#include <ponos.h>
#include <poseidon.h>
#include <sstream>
#include <iomanip>
using namespace aergia;
using namespace ponos;
using namespace poseidon;

int WIDTH = 800, HEIGHT = 800;
SceneApp<> app(WIDTH, HEIGHT, "Hello Grid Solver!", false);
GridSolver2D solver;

int main() {
  WIN32CONSOLE();
  BBox2D region(Point2(), Point2(1.f));
  solver.set(10, region);
  auto grid = solver.getGrid();
  auto scene = solver.getScene();
  scene.addCollider(Collider2D(new ImplicitCircle(Point2(0.5f), 0.25)));
  // scene.addCollider(Collider2D(new ImplicitPlane2D(Point2(), Normal2D(0,
  // 1))));
  RawMesh rm;
  marchingSquares(scene.getSDF(), region, 20, 20, &rm);
  app.init();
  app.addViewport2D(0, 0, WIDTH, HEIGHT);
  app.getCamera<Camera2D>(0)->fit(region, 1.1f);
  app.scene.add(new SceneMesh(&rm, [&](Shader *s) { glColor4f(1, 0, 1, 1); }));
  app.scene.add(new StaggeredGrid2DModel<StaggeredGrid2f>(&grid));
  auto sdf = new GridModel<ScalarGrid2f>(scene.getSDF());
  Text *text = new Text("C:/Windows/Fonts/Arial.ttf");
  sdf->f = [&](const float &v, Point3 p) {
    // vec2f ve = scene.getSDF()->gradient(p.x, p.y);
    // draw_vector(Point2(p.x, p.y), 0.04 * vec2(ve[0], ve[1]));
    std::ostringstream stringStream;
    stringStream << std::setprecision(2) << scene.getSDF()->sample(p.x, p.y);
    std::string copyOfStr = stringStream.str();
    text->render(copyOfStr, glGetMVPTransform()(p), 0.2f, Color(0, 0, 0, 1));
  };
  grid.u.setAll(1.f);
  solver.enforceBoundaries();
  auto sdf2 = new GridModel<ScalarGrid2f>(new ScalarGrid2f(100, 100, region));
  sdf2->gridColor = COLOR_TRANSPARENT;
  sdf2->f = [&](const float &v, Point3 p) {
    float ve = scene.getSDF()->sample(p.x, p.y);
    if (ve > 0.0)
      glColor4f(ve, 0, 0, 1.f);
    else
      glColor4f(0, 0, ve + 0.5, 1.f);
    glPointSize(3.f);
    glBegin(GL_POINTS);
    glVertex(p);
    glEnd();
  };
  app.scene.add(sdf);
  app.scene.add(sdf2);
  app.run();
  return 0;
}
