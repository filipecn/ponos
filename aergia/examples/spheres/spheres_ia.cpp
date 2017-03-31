#include <aergia.h>
#include <ponos.h>
#include "interval_method.h"

using namespace aergia;

int WIDTH = 1200, HEIGHT = 800;

int bot_left[4] = {0, 0, WIDTH / 3, HEIGHT / 2};
int top_left[4] = {0, HEIGHT / 2, WIDTH / 3, HEIGHT / 2};
int ia_view[4] = {WIDTH / 3, 0, WIDTH / 3, HEIGHT};
int aa_view[4] = {2 * WIDTH / 3, 0, WIDTH / 3, HEIGHT};

SceneApp<> app(WIDTH, HEIGHT, "Spheres IA", false);
ViewportDisplay iaView(ia_view[0], ia_view[1], ia_view[2], ia_view[3]);
ViewportDisplay aaView(aa_view[0], aa_view[1], aa_view[2], aa_view[3]);

class SphereObject : public aergia::SceneObject {
public:
  SphereObject(ponos::Sphere s, float r, float g, float b, int _id) {
    sphere.c = s.c;
    sphere.r = s.r;
    rgb[0] = r;
    rgb[1] = g;
    rgb[2] = b;
    rgb[3] = 0.3;
    id = _id;
  }

  void draw() const override {
    glColor4fv(rgb);
    if (this->selected)
      glColor4f(rgb[0], rgb[1], rgb[2], 0.5);
    draw_sphere(sphere, &this->transform);
  }

  bool intersect(const ponos::Ray3 &r, float *t = nullptr) override {
    ponos::Sphere ts;
    ts.c = transform(sphere.c);
    ts.r = ponos::distance(
        ts.c, transform(ponos::Point3(sphere.r, 0, 0) + ponos::vec3(sphere.c)));
    return ponos::sphere_ray_intersection(ts, r, t);
  }

  ponos::Sphere sphere;
  float rgb[4];
};

template <typename Interval> class CollisionTester {
public:
  struct Node {
    Node(Interval T, Interval P) {
      PHI = P;
      THETA = T;
      for (int i = 0; i < 4; i++)
        child[i] = nullptr;
      color = 0;
      id = -1;
      frame = -1;
      isLeaf = false;
    }
    void set(char c, int i, int f) {
      color = c;
      id = i;
      frame = f;
    }
    char color;
    int id;
    int frame;
    bool isLeaf;
    Node *child[4];
    Interval PHI, THETA;
  };

  CollisionTester(int l) : maxLevel(l) {
    root = new Node(Interval(0.0, PI_2), Interval(0.0, PI));
    curFrame = 0;
  }
  void setSphere(ponos::Sphere c, ponos::Transform t) {
    s.c = t(c.c);
    s.r = ponos::distance(t(ponos::Point3(c.r, 0, 0)), s.c);
  }
  void addSphere(ponos::Sphere ss, ponos::Transform t) {
    sj.c = t(ss.c);
    sj.r = ponos::distance(t(ponos::Point3(ss.r, 0, 0)), sj.c);
    refine(root, 0);
  }
  void draw() { draw(root); }

  int curFrame, curId;
  ponos::Sphere s, sj;

private:
  void refine(Node *n, int level) {
    if (level >= maxLevel)
      return;
    if (curFrame == n->frame && curId == n->id && n->color == 2)
      return;
    n->isLeaf = false;
    if (level == maxLevel - 1)
      n->isLeaf = true;
    Interval I = square(static_cast<double>(s.r) * cos(n->THETA) * sin(n->PHI) +
                        static_cast<double>(s.c.x - sj.c.x)) +
                 square(static_cast<double>(s.r) * sin(n->THETA) * sin(n->PHI) +
                        static_cast<double>(s.c.y - sj.c.y)) +
                 square(static_cast<double>(s.r) * cos(n->PHI) +
                        static_cast<double>(s.c.z - sj.c.z)) -
                 static_cast<double>(SQR(sj.r));
    if (I.upper() < 0.0) {
      n->isLeaf = true;
      n->set(2, curId, curFrame);
      return;
    }
    if (zero_in(I)) {
      n->set(1, curId, curFrame);
      Interval thetaIntervals[4] = {
          Interval(n->THETA.lower(), n->THETA.lower() + width(n->THETA) / 2.0),
          Interval(n->THETA.lower() + width(n->THETA) / 2.0,
                   n->THETA.lower() + width(n->THETA)),
          Interval(n->THETA.lower(), n->THETA.lower() + width(n->THETA) / 2.0),
          Interval(n->THETA.lower() + width(n->THETA) / 2.0,
                   n->THETA.lower() + width(n->THETA))};
      Interval phiIntervals[4] = {
          Interval(n->PHI.lower(), n->PHI.lower() + width(n->PHI) / 2.0),
          Interval(n->PHI.lower(), n->PHI.lower() + width(n->PHI) / 2.0),
          Interval(n->PHI.lower() + width(n->PHI) / 2.0,
                   n->PHI.lower() + width(n->PHI)),
          Interval(n->PHI.lower() + width(n->PHI) / 2.0,
                   n->PHI.lower() + width(n->PHI))};
      for (int i = 0; i < 4; i++) {
        if (!n->child[i])
          n->child[i] = new Node(thetaIntervals[i], phiIntervals[i]);
        refine(n->child[i], level + 1);
      }
      int mixColors = -1;
      if (n->child[0]->color == n->child[1]->color &&
          n->child[2]->color == n->child[3]->color &&
          n->child[0]->color == n->child[3]->color)
        mixColors = n->child[0]->color;
      if (level == maxLevel - 1 || mixColors == 2) {
        if (mixColors == 2)
          n->set(mixColors, curId, curFrame);
        n->isLeaf = true;
      }
      return;
    }
    if (n->frame != curFrame || n->id != curId ||
        (n->frame == curFrame && n->id == curId && n->color == 0)) {
      n->set(0, curId, curFrame);
      n->isLeaf = true;
    }
  }

  void draw(Node *n) {
    if (!n)
      return;
    if (curFrame != n->frame || curId != n->id)
      return;
    glColor3f(0, 0, 0);
    float *c = nullptr;
    float co[4] = {1, 0, 0, 0.3};
    float coi[4] = {0, 0, 0, 0.3};
    if (n->color == 1)
      c = co;
    else if (n->color == 2)
      c = coi;
    if (n->isLeaf) {
      aergia::draw_bbox(
          ponos::BBox2D(ponos::Point2(n->PHI.lower(), n->THETA.lower()),
                        ponos::Point2(n->PHI.upper(), n->THETA.upper())),
          c);
      return;
    }
    for (int i = 0; i < 4; i++)
      draw(n->child[i]);
  }
  Node *root;
  int maxLevel;
};

CollisionTester<IAI> iaTester(6);
CollisionTester<AAI> aaTester(6);

void renderIATree() {
  iaTester.curFrame++;
  iaTester.curId = 0;
  app.scene.iterateObjects([](const SceneObject *o) {
    if (o->id == 0)
      iaTester.setSphere(static_cast<const SphereObject *>(o)->sphere,
                         o->transform);
  });
  app.scene.iterateObjects([](const SceneObject *o) {
    if (o->id > 0)
      iaTester.addSphere(static_cast<const SphereObject *>(o)->sphere,
                         o->transform);
  });
  iaTester.draw();
}

void renderAATree() {
  return;
  aaTester.curFrame++;
  aaTester.curId = 0;
  app.scene.iterateObjects([](const SceneObject *o) {
    if (o->id == 0)
      aaTester.setSphere(static_cast<const SphereObject *>(o)->sphere,
                         o->transform);
  });
  app.scene.iterateObjects([](const SceneObject *o) {
    if (o->id > 0)
      aaTester.addSphere(static_cast<const SphereObject *>(o)->sphere,
                         o->transform);
  });
  aaTester.draw();
}

void render() {
  iaView.render();
  aaView.render();
}

int main(int argc, char **argv) {
#ifdef _WIN32
  WIN32CONSOLE();
#endif
  iaView.camera.reset(new aergia::Camera2D());
  static_cast<aergia::Camera2D *>(iaView.camera.get())
      ->resize(ia_view[2], ia_view[3]);
  static_cast<aergia::Camera2D *>(iaView.camera.get())
      ->setPos(ponos::vec2(PI / 2.0, PI_2 / 2.0));
  static_cast<aergia::Camera2D *>(iaView.camera.get())->setZoom(1.3);
  iaView.renderCallback = renderIATree;
  aaView.camera.reset(new aergia::Camera2D());
  static_cast<aergia::Camera2D *>(aaView.camera.get())
      ->resize(aa_view[2], aa_view[3]);
  static_cast<aergia::Camera2D *>(aaView.camera.get())
      ->setPos(ponos::vec2(PI / 2.0, PI_2 / 2.0));
  static_cast<aergia::Camera2D *>(aaView.camera.get())->setZoom(1.3);
  aaView.renderCallback = renderAATree;
  app.renderCallback = render;
  // bottom left window
  app.addViewport(bot_left[0], bot_left[1], bot_left[2], bot_left[3]);
  static_cast<Camera *>(app.viewports[0].camera.get())
      ->setPos(ponos::Point3(10, 0, 0));
  // top left window
  app.addViewport(top_left[0], top_left[1], top_left[2], top_left[3]);
  static_cast<Camera *>(app.viewports[1].camera.get())
      ->setPos(ponos::Point3(0, 10, 0));
  static_cast<Camera *>(app.viewports[1].camera.get())
      ->setUp(ponos::vec3(1, 0, 0));
  app.init();
  app.scene.add(new aergia::CartesianGrid(5, 5, 5));
  app.scene.add(
      new SphereObject(ponos::Sphere(ponos::Point3(), 0.8f), 0, 0, 0.5, 1));
  app.scene.add(
      new SphereObject(ponos::Sphere(ponos::Point3(), 0.5f), 1, 0, 0.5, 0));
  app.scene.add(
      new SphereObject(ponos::Sphere(ponos::Point3(), 0.6f), 0, 0, 0.5, 2));
  app.scene.add(
      new SphereObject(ponos::Sphere(ponos::Point3(), 1.2f), 0, 0, 0.5, 3));
  app.scene.add(
      new SphereObject(ponos::Sphere(ponos::Point3(), 0.7f), 0, 0, 0.5, 4));
  app.run();
  return 0;
}
