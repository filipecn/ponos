#include <aergia/aergia.h>
#include <ponos/ponos.h>

using namespace aergia;

int WIDTH = 800, HEIGHT = 800;

SceneApp<> app(WIDTH, HEIGHT, "Stream Lines");
ponos::BBox bbox(ponos::Point3(), ponos::Point3(1, 1, 1));
ponos::CRegularGrid<ponos::vec3> grid(ponos::ivec3(15), ponos::vec3(), bbox);

class VectorsGrid : public SceneObject {
public:
  VectorsGrid() {}
  /* @inherit */
  void draw() override {
    glPointSize(3);
    glBegin(GL_POINTS);
    ponos::ivec3 ijk;
    FOR_INDICES0_3D(grid.dimensions, ijk) {
      ponos::vec3 s = grid(ijk[0], ijk[1], ijk[2]);
      glColor4f(s.x, s.y, s.z, 1);
      ponos::Point3 p = grid.toWorld(ponos::Point3(ijk[0], ijk[1], ijk[2]));
      glVertex(p);
    }
    glEnd();
  }
};

class RandomPoints : public SceneObject {
public:
  RandomPoints(int size) {
    rng[0] = new ponos::HaltonSequence(2);
    rng[1] = new ponos::HaltonSequence(3);
    rng[2] = new ponos::HaltonSequence(5);
    while (size) {
      ponos::Point3 p(rng[0]->randomFloat(), rng[1]->randomFloat(),
                      rng[2]->randomFloat());
      points.emplace_back(p);
      size--;
    }
  }

  void draw() override {
    glPointSize(4);
    glBegin(GL_POINTS);
    for (uint i = 0; i < points.size(); i++) {
      ponos::vec3 c = grid(ponos::vec3(points[i]));
      glColor4f(c.x, c.y, c.z, 0.8);
      aergia::glVertex(points[i]);
    }
    glEnd();
  }

  std::vector<ponos::Point3> points;

private:
  ponos::HaltonSequence *rng[3];
};

int main() {
  app.init();

  ponos::ivec3 ijk;
  FOR_INDICES0_3D(grid.dimensions, ijk) {
    ponos::vec3 v = ponos::vec3(ijk[0], ijk[1], ijk[2]) / 15.f;
    grid.set(ijk, v);
  }

  app.scene.add(new VectorsGrid());
  app.scene.add(new RandomPoints(100));
  app.run();
  return 0;
}
