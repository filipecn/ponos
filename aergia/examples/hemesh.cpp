#include <aergia.h>
#include <ponos.h>

aergia::SceneApp<> app(800, 800, "HEMesh Example", false);
aergia::Text *text;

class HEMeshObject : public aergia::SceneObject {
public:
  HEMeshObject(const ponos::RawMesh *rm) {
    mesh.reset(new ponos::HEMesh<float, 2>(rm));
    mesh->addEdge(0, 1);
    mesh->addEdge(4, 0);
    mesh->addEdge(1, 4);
    mesh->addEdge(4, 3);
    mesh->addEdge(0, 3);
    mesh->addEdge(1, 2);
    // mesh->addEdge(1, 5);
    // mesh->traverseEdgesFromVertex(0,
    //                              [](int e) { std::cout << e << std::endl; });
    // std::cout << std::endl;
    // mesh->traverseEdgesToVertex(0, [](int e) { std::cout << e << std::endl;
    // });
  }

  void draw() const override {
    const std::vector<ponos::HEMesh<float, 2>::Vertex> &vertices =
        mesh->getVertices();
    const std::vector<ponos::HEMesh<float, 2>::Edge> &edges = mesh->getEdges();
    const std::vector<ponos::HEMesh<float, 2>::Face> &faces = mesh->getFaces();
    glPointSize(4.f);
    glColor4f(0, 0, 0, 0.5);
    glBegin(GL_POINTS);
    for (auto p : vertices)
      aergia::glVertex(p.position);
    glEnd();
    for (auto e : edges) {
      glColor4f(0, 0, 0, 0.5);
      ponos::Point2 a = vertices[e.orig].position.floatXY();
      ponos::Point2 b = vertices[e.dest].position.floatXY();
      ponos::vec2 v = ponos::normalize(b - a);
      glBegin(GL_LINES);
      aergia::glVertex(a);
      aergia::glVertex(b);
      glEnd();
      glColor4f(1, 0, 0, 0.9);
      ponos::Point2 A = a + 0.03f * v.left() + 0.15f * v;
      ponos::Point2 B = b + 0.03f * v.left() - 0.15f * v;
      aergia::draw_vector(A, B - A);
      glColor4f(1, 0, 0, 0.4);
      glBegin(GL_LINES);
      if (e.next >= 0) {
        aergia::glVertex(B);
        ponos::vec2 V =
            ponos::normalize(vertices[edges[e.next].dest].position.floatXY() -
                             vertices[edges[e.next].orig].position.floatXY());
        aergia::glVertex(vertices[edges[e.next].orig].position.floatXY() +
                         0.03f * V.left() + 0.15f * V);
      }
      if (e.prev >= 0) {
        aergia::glVertex(A);
        ponos::vec2 V =
            ponos::normalize(vertices[edges[e.prev].dest].position.floatXY() -
                             vertices[edges[e.prev].orig].position.floatXY());
        aergia::glVertex(vertices[edges[e.prev].dest].position.floatXY() +
                         0.03f * V.left() - 0.15f * V);
      }
      glEnd();
    }
    glColor4f(0, 1, 0, 0.3);
    glBegin(GL_TRIANGLES);
    for (size_t i = 0; i < faces.size(); i++) {
      mesh->traversePolygonEdges(i, [&edges, &vertices](int e) {
        aergia::glVertex(vertices[edges[e].orig].position);
      });
    }
    glEnd();
    text->render("This is !xt", 25.0f, 25.0f, 1.0f,
                 aergia::Color(0.5, 0.8f, 0.2f));
  }

private:
  std::shared_ptr<ponos::HEMesh<float, 2>> mesh;
};

int main() {
#ifdef WIN32
  WIN32CONSOLE();
#endif
  ponos::RawMesh rm;
  rm.meshDescriptor.elementSize = 3;
  rm.meshDescriptor.count = 0;
  rm.vertexDescriptor.elementSize = 2;
  rm.vertexDescriptor.count = 6;
  rm.addVertex({0, 1});
  rm.addVertex({1, 1});
  rm.addVertex({2, 1});
  rm.addVertex({0, 0});
  rm.addVertex({1, 0});
  rm.addVertex({2, 0});
  // rm.addFace({{0, 0, 0}, {3, 0, 0}, {4, 0, 0}});
  // rm.addFace({{0, 0, 0}, {4, 0, 0}, {1, 0, 0}});
  // rm.addFace({{1, 0, 0}, {4, 0, 0}, {5, 0, 0}});
  // rm.addFace({{1, 0, 0}, {5, 0, 0}, {2, 0, 0}});
  rm.computeBBox();
  std::cout << rm.bbox.pMin << rm.bbox.pMax << std::endl;
  app.init();
  app.addViewport2D(0, 0, 800, 800);
  static_cast<aergia::Camera2D *>(app.viewports[0].camera.get())
      ->fit(ponos::BBox2D(rm.bbox.pMin.xy(), rm.bbox.pMax.xy()), 1.1f);
  app.scene.add(new HEMeshObject(&rm));
  text = new aergia::Text("C:/Windows/Fonts/Arial.ttf");
  app.run();
  return 0;
}
