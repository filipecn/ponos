#include <aergia.h>
#include <ponos.h>

aergia::SceneApp<> app(800, 800, "HEMesh Example", false);
aergia::Text *text;

class HEMeshObject : public aergia::SceneObject {
public:
  HEMeshObject(const ponos::RawMesh *rm) {
    mesh.reset(new ponos::HEMesh<float, 2>(rm));
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
    for (size_t k = 0; k < vertices.size(); k++) {
      char label[100];
      sprintf(label, "%d", k);
      ponos::Point3 labelPosition = app.viewports[0].camera->getTransform()(
          ponos::Point3(vertices[k].position[0], vertices[k].position[1], 0));
      text->render(label, labelPosition, .5f, aergia::Color(0.8f, 0.2f, 0.7f));
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
      glColor4f(1, 0, 0, 0.9);
      ponos::Point2 A = a + 0.03f * v.left() + 0.15f * v;
      ponos::Point2 B = b + 0.03f * v.left() - 0.15f * v;
      ponos::Point2 C = A + (B - A) / 2.f;
      ponos::Point3 labelPositition =
          app.viewports[0].camera->getTransform()(ponos::Point3(C.x, C.y, 0.f));
      char label[100];
      sprintf(label, "%d", k++);
      text->render(label, labelPositition, .3f,
                   aergia::Color(0.4f, 0.2f, 0.7f));
      aergia::draw_vector(A, B - A, 0.04, 0.05);
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
    for (size_t i = 0; i < faces.size(); i++) {
      ponos::Point2 mp(0, 0);
      glColor4f(0, 1, 0, 0.3);
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
      sprintf(label, "%d", i);
      text->render(label, labelPosition, .5f, aergia::Color(0.8f, 0.5f, 0.2f));
    }
  }

private:
  std::shared_ptr<ponos::HEMesh<float, 2>> mesh;
};

int main() {
#ifdef WIN32
  WIN32CONSOLE();
#endif
  ponos::RawMesh rm;
  aergia::loadPLY("C:/Users/fuiri/Desktop/2d.tar/2d/PLY/circle.ply", &rm);
  std::cout << rm.bbox.pMin << rm.bbox.pMax << std::endl;
  app.init();
  app.addViewport2D(0, 0, 800, 800);
  static_cast<aergia::Camera2D *>(app.viewports[0].camera.get())
      ->fit(ponos::BBox2D(rm.bbox.pMin.xy(), rm.bbox.pMax.xy()), 1.1f);
  app.scene.add(new HEMeshObject(&rm));
  text = new aergia::Text("C:/Windows/Fonts/Arial.ttf");
  app.scrollCallback = [](double dx, double dy) {
    static float z = 1.f;
    z *= (dy < 0.f) ? 0.9f : 1.1f;
    static_cast<aergia::Camera2D *>(app.viewports[0].camera.get())->setZoom(z);
    ponos::vec2 p = static_cast<aergia::Camera2D *>(
                        app.viewports[0].camera.get())->getPos();
  };
  app.run();
  return 0;
}
