#include <aergia/aergia.h>

int WIDTH = 800, HEIGHT = 800;

aergia::SceneApp<> app(WIDTH, HEIGHT, "Queries");

ponos::Ray2 r(ponos::Point2(0.147905, 0.0100919),
              ponos::vec2(0.280415, -0.959));
ponos::BBox2D bb(ponos::Point2(0.14, 0.0), ponos::Point2(0.15, 0.01));

void render() {
  glColor4f(0, 0, 0, 0.1);
  aergia::draw_bbox(bb);
  float t1, t2;
  if (ponos::bbox_ray_intersection(bb, r, t1, t2))
    glColor3f(1, 0, 0);
  else
    glColor3f(0, 0, 1);
  glBegin(GL_LINES);
  aergia::glVertex(r.o);
  // aergia::glVertex(r.o + 0.001 * r.d);
  aergia::glVertex(ponos::Point2(0.147792, 0.0099473));
  glEnd();
}

int main() {
  app.viewports[0].renderCallback = render;
  app.viewports[0].camera.reset(new aergia::Camera2D());
  static_cast<aergia::Camera2D *>(app.viewports[0].camera.get())
      ->resize(WIDTH / 2, HEIGHT);
  static_cast<aergia::Camera2D *>(app.viewports[0].camera.get())
      ->setPos(ponos::vec2(bb.pMin[0] + bb.size(0) * 0.5f,
                           bb.pMin[1] + bb.size(1) * 0.5f));
  static_cast<aergia::Camera2D *>(app.viewports[0].camera.get())->setZoom(0.1f);
  app.init();
  app.run();
  return 0;
}
