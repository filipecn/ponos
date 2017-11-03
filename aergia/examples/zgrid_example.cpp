#include <aergia/aergia.h>
#include <ponos/ponos.h>

using namespace aergia;

SceneApp<> app(800, 800, "ZGrid Example");
ponos::CZGrid<float> color[2];

class ZGridObject : public SceneObject {
public:
  ZGridObject() {}
  void draw() const override {
    ponos::ivec2 ij;
    glPointSize(5);
    glBegin(GL_POINTS);
    FOR_INDICES0_2D(color[0].getDimensions(), ij) {
      glColor4f(color[0](ij), color[1](ij), 0, 1.f);
      glVertex(color[0].dataWorldPosition(ij));
    }
    glEnd();
  }
};

class RandomPoints : public SceneObject {
public:
  RandomPoints(int size) {
    rng[0] = new ponos::HaltonSequence(2);
    rng[1] = new ponos::HaltonSequence(3);
    while (size) {
      ponos::Point2 p(rng[0]->randomFloat(), rng[1]->randomFloat());
      points.emplace_back(p);
      size--;
    }
  }

  void draw() const override {
    glPointSize(5.f);
    glBegin(GL_POINTS);
    for (uint i = 0; i < points.size(); i++) {
      float rgb = color[0].sample(points[i].x, points[i].y);
      float rgb2 = color[1].sample(points[i].x, points[i].y);
      glColor4f(rgb, rgb2, 0, 0.8);
      aergia::glVertex(points[i]);
    }
    glEnd();
  }

  std::vector<ponos::Point2> points;

private:
  ponos::HaltonSequence *rng[2];
};

int main() {
  app.init();
  for (int i = 0; i < 2; i++) {
    color[i].set(16, 16, ponos::vec2(0.f), 1.f / 16.f);
  }
  ponos::ivec2 ij;
  FOR_INDICES0_2D(color[0].getDimensions(), ij) {
    color[0](ij) = static_cast<float>(ij[0]) / 16.f;
    color[1](ij) = static_cast<float>(ij[1]) / 16.f;
  }
  app.scene.add(new ZGridObject());
  app.scene.add(new RandomPoints(100));
  app.run();
  return 0;
}
