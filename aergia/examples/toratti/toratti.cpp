#include <aergia.h>
#include <ponos.h>

aergia::SceneApp<> app(800, 800, "HEMesh Example", false);
aergia::Text *text;

class HEMeshObject : public aergia::SceneObject {
public:
  HEMeshObject(const ponos::RawMesh *rm) {
    mesh.reset(new ponos::HEMesh2DF(rm));
    std::vector<size_t> p(1, 0);
    ponos::fastMarch2D(mesh.get(), p);
  }

  void draw() const override {
    const std::vector<ponos::HEMesh2DF::Vertex> &vertices = mesh->getVertices();
    const std::vector<ponos::HEMesh2DF::Edge> &edges = mesh->getEdges();
    const std::vector<ponos::HEMesh2DF::Face> &faces = mesh->getFaces();
    glPointSize(4.f);
    glColor4f(0, 0, 0, 0.5);
    glBegin(GL_POINTS);
    for (auto p : vertices)
      aergia::glVertex(p.position);
    glEnd();
    for (size_t k = 0; k < vertices.size(); k++) {
      char label[100];
      sprintf(label, "%lu", k);
      ponos::Point3 labelPosition = app.viewports[0].camera->getTransform()(
          ponos::Point3(vertices[k].position[0], vertices[k].position[1], 0));
      text->render(label, labelPosition, .5f,
                   aergia::Color(0.8f, 0.2f, 0.7f, 0.1f));
      sprintf(label, " %f", vertices[k].data);
      text->render(label, labelPosition, .3f, aergia::Color(1.0f, 0.2f, 0.4f));
    }
    int k = 0;
    for (auto e : edges) {
      glColor4f(0, 0, 0, 0.5);
      ponos::Point2 a = vertices[e.orig].position.floatXY();
      ponos::Point2 b = vertices[e.dest].position.floatXY();
      ponos::vec2 v = ponos::normalize(b - a);
      glBegin(GL_LINES);
      aergia::glVertex(a);
      aergia::glVertex(b);
      glEnd();
      glColor4f(1, 0, 0, 0.3);
      ponos::Point2 A = a + 0.03f * v.left() + 0.15f * v;
      ponos::Point2 B = b + 0.03f * v.left() - 0.15f * v;
      ponos::Point2 C = A + (B - A) / 2.f;
      ponos::Point3 labelPositition =
          app.viewports[0].camera->getTransform()(ponos::Point3(C.x, C.y, 0.f));
      char label[100];
      sprintf(label, "%d", k++);
      text->render(label, labelPositition, .3f,
                   aergia::Color(0.4f, 0.2f, 0.7f, 0.3f));
      aergia::draw_vector(A, B - A, 0.04, 0.05);
      glColor4f(1, 0, 0, 0.2);
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
    for (size_t i = 0; i < faces.size(); i++) {
      ponos::Point2 mp(0, 0);
      glColor4f(0, 1, 0, 0.1);
      glBegin(GL_TRIANGLES);
      mesh->traversePolygonEdges(i, [&mp, &edges, &vertices](int e) {
        aergia::glVertex(vertices[edges[e].orig].position);
        mp[0] += vertices[edges[e].orig].position[0];
        mp[1] += vertices[edges[e].orig].position[1];
      });
      glEnd();
      char label[100];
      ponos::Point3 labelPosition = app.viewports[0].camera->getTransform()(
          ponos::Point3(mp[0] / 3.f, mp[1] / 3.f, 0.f));
      sprintf(label, "%lu", i);
      text->render(label, labelPosition, .5f,
                   aergia::Color(0.8f, 0.5f, 0.2f, 0.2f));
    }
  }

private:
  std::shared_ptr<ponos::HEMesh2DF> mesh;
};

int main() {
  WIN32CONSOLE();
  ponos::RawMesh rm;
  // aergia::loadPLY("C:/Users/fuiri/Desktop/2d.tar/2d/PLY/circle.ply", &rm);
  aergia::loadPLY("/mnt/c/Users/fuiri/Desktop/2d.tar/2d/PLY/circle.ply", &rm);
  std::cout << rm.bbox.pMin << rm.bbox.pMax << std::endl;
  app.init();
  app.addViewport2D(0, 0, 800, 800);
  app.getCamera<aergia::Camera2D>(0)
      ->fit(ponos::BBox2D(rm.bbox.pMin.xy(), rm.bbox.pMax.xy()), 1.1f);
  app.scene.add(new HEMeshObject(&rm));
  text = new aergia::Text("/mnt/c/Windows/Fonts/arial.ttf");
  app.scrollCallback = [](double dx, double dy) {
    static float z = 1.f;
    z *= (dy < 0.f) ? 0.9f : 1.1f;
    app.getCamera<aergia::Camera2D>(0)->setZoom(z);
    ponos::vec2 p = app.getCamera<aergia::Camera2D>(0)->getPos();
  };
  // app.run();
  return 0;
}
