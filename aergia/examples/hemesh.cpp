#include <aergia/aergia.h>
#include <ponos/ponos.h>

aergia::SceneApp<> app(800, 800, "HEMesh Example", false);

int main() {
  WIN32CONSOLE();
  ponos::RawMesh rm;
  aergia::loadPLY(
      "/run/media/filipecn/OS/Users/fuiri/Desktop/2d.tar/2d/PLY/circle.ply",
      &rm);
  std::cout << rm.bbox.pMin << rm.bbox.pMax << std::endl;
  app.init();
  app.addViewport2D(0, 0, 800, 800);
  app.getCamera<aergia::UserCamera2D>(0)->fit(
      ponos::BBox2D(rm.bbox.pMin.xy(), rm.bbox.pMax.xy()), 1.1f);
  app.scene.add(new aergia::HEMeshObject(
      new ponos::HEMesh2DF(&rm),
      new aergia::TextRenderer("/run/media/filipecn/OS/Windows/Fonts/arial.ttf")));
  app.scrollCallback = [](double dx, double dy) {
    UNUSED_VARIABLE(dx);
    static float z = 1.f;
    z *= (dy < 0.f) ? 0.9f : 1.1f;
    app.getCamera<aergia::UserCamera2D>(0)->setZoom(z);
  };
  app.run();
  return 0;
}
