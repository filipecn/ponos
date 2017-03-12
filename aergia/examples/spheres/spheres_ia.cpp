#include <aergia.h>
#include <ponos.h>

using namespace aergia;

int WIDTH = 1200, HEIGHT = 800;

SceneApp<> app(WIDTH, HEIGHT, "Spheres IA", false);
ViewportDisplay parametricView(WIDTH / 3, 0, 2 * WIDTH / 3, HEIGHT);

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

void render() { parametricView.render(); }

int main(int argc, char **argv) {
  // WIN32CONSOLE();
  parametricView.camera.reset(new aergia::Camera2D());
  app.renderCallback = render;
  // bottom left window
  app.addViewport(0, 0, WIDTH / 3, HEIGHT / 2);
  static_cast<Camera *>(app.viewports[0].camera.get())
      ->setPos(ponos::Point3(10, 0, 0));
  // top left window
  app.addViewport(0, HEIGHT / 2, WIDTH / 3, HEIGHT / 2);
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
  app.scene.iterateObjects(
      [](const SceneObject *o) { std::cout << o->id << std::endl; });
  app.run();
  return 0;
}
