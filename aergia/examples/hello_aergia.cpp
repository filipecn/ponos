#include <aergia.h>
#include <iostream>
#include <ponos.h>
#include <vector>

aergia::GraphicsDisplay &gd = aergia::GraphicsDisplay::instance();
aergia::Camera2D camera;
ponos::Transform toWorld;
ponos::Brep<float, 2> brep;
ponos::BBox2D bbox;
enum modes { VERTEX = 0, EDGE, FACE, MODES };
char mode = 0;
int sVertex = -1;
int sA = -1;

ponos::Point2 worldMousePoint() {
  ponos::Point2 mp = gd.getMouseNPos();
  return toWorld(ponos::Point3(mp.x, mp.y, 1.f)).xy();
}

int selectVertex(ponos::Point2 wp) {
  for (size_t i = 0; i < brep.vertices.size(); ++i)
    if (ponos::distance(wp, brep.vertices[i].pos.floatXY()) < 0.1f)
      return static_cast<int>(i);
  return -1;
}

void button(int button, int action) {
  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
    mode = (mode + 1) % MODES;
    return;
  }
  ponos::Point2 wp = worldMousePoint();
  if (action == GLFW_RELEASE) {
    switch (mode) {
    case VERTEX: {
      sVertex = selectVertex(wp);
      if (sVertex < 0)
        sVertex = brep.addVertex(ponos::Point<float, 2>(wp));
    } break;
    case EDGE: {
      sVertex = selectVertex(wp);
      if (sVertex < 0)
        sVertex = brep.addVertex(ponos::Point<float, 2>(wp));
      if (sA != sVertex)
        brep.addEdge(sA, sVertex);
      sA = sVertex = -1;
    } break;
    }
  } else {
    switch (mode) {
    case VERTEX: {
    } break;
    case EDGE: {
      sA = selectVertex(wp);
      if (sA < 0)
        sA = brep.addVertex(ponos::Point<float, 2>(wp));
    } break;
    }
  }
}

void mouse(double x, double y) {
  UNUSED_VARIABLE(x);
  UNUSED_VARIABLE(y);
  if (sVertex >= 0) {
    // flip.particleGrid.setPos(sVertex, worldMousePoint());
    // ponos::vec2 d = flip.particleGrid.getParticle(1).p -
    // flip.particleGrid.getParticle(0).p;
    // d /= flip.dt;
    // flip.particleGrid.getParticleReference(0).v = d;
    // flip.particleGrid.setPos(2,
    // flip.newPosition(flip.particleGrid.getParticle(0)));
  }
}

void drawVertices() {
  glPointSize(8);
  glBegin(GL_POINTS);
  for (size_t i = 0; i < brep.vertices.size(); i++) {
    glColor3f(0, 0, 0);
    if (sVertex == static_cast<int>(i))
      glColor3f(1, 0, 0);
    aergia::glVertex(brep.vertices[i].pos.floatXY());
  }
  glEnd();
}

void drawEdges() {
  glBegin(GL_LINES);
  if (sA >= 0) {
    ponos::Point2 wp = worldMousePoint();
    glColor3f(0, 0, 0);
    aergia::glVertex(wp);
    aergia::glVertex(brep.vertices[sA].pos.floatXY());
  }
  for (size_t i = 0; i < brep.edges.size(); i++) {
    glColor3f(0, 0, 0);
    ponos::Point2 a = brep.vertices[brep.edges[i].a].pos.floatXY();
    ponos::Point2 b = brep.vertices[brep.edges[i].b].pos.floatXY();
    ponos::Point2 mp = a + (b - a) / 2.f;
    aergia::glVertex(a);
    aergia::glVertex(b);
    aergia::glVertex(mp + orthonormal(b - a, false));
    aergia::glVertex(mp + (b - a) / 4.f);
    aergia::glVertex(mp + orthonormal(b - a, true));
    aergia::glVertex(mp + (b - a) / 4.f);
  }
  glEnd();
}

void drawFaces() {}

void render() {
  gd.clearScreen(1.f, 1.f, 1.f, 0.f);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  camera.look();
  drawVertices();
  drawEdges();
  drawFaces();
}

void loadMesh() {
  int nv = 0;
  std::cin >> nv;
  for (int i = 0; i < nv; i++) {
    ponos::Brep<float, 2>::Vertex v;
    // std::cin >> v.pos[0] >> v.pos[1] >> v.edge;
    brep.addVertex(v);
    bbox = ponos::make_union(bbox, ponos::Point2(v.pos[0], v.pos[1]));
  }
  int ne = 0;
  std::cin >> ne;
  for (int i = 0; i < ne; i++) {
    ponos::Brep<float, 2>::Edge e;
    std::cin >> e.a >> e.b;
    brep.addEdge(e);
  }
  int nf = 0;
  std::cin >> nf;
  while (nf--) {
    int nvs = 0;
    std::cin >> nvs;
    std::vector<int> vs(nvs);
    for (int i = 0; i < nvs; i++) {
      std::cin >> vs[i];
    }
    brep.addFace(vs);
  }
}

int main() {
  bbox = ponos::make_union(bbox, ponos::Point2(-5, -5));
  bbox = ponos::make_union(bbox, ponos::Point2(-5, 5));
  bbox = ponos::make_union(bbox, ponos::Point2(5, 5));
  bbox = ponos::make_union(bbox, ponos::Point2(5, -5));
  // loadMesh();
  // set camera
  camera.setPos(ponos::vec2(bbox.center().x, bbox.center().y));
  camera.setZoom((bbox.pMax - bbox.pMin)[0]);
  camera.resize(800, 800);
  toWorld = ponos::Transform(ponos::inverse(camera.getTransform()));
  // init window
  aergia::createGraphicsDisplay(800, 800, "FLIP - 2D");
  gd.registerRenderFunc(render);
  gd.registerButtonFunc(button);
  gd.registerMouseFunc(mouse);
  gd.start();
  return 0;
}
