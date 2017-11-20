#include <aergia.h>
#include <ponos.h>
#include <cstring>
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

ponos::Timer cl;
std::vector<ponos::Point2> positions;
char nameFormat[100];
int curFrame;

bool readFrame(int f) {
  char filename[100];
  sprintf(filename, nameFormat, f);
  std::cout << "reading " << filename << std::endl;
  FILE *fp = fopen(filename, "r");
  if (!fp)
    return false;
  positions.clear();
  float x, y;
  while (fscanf(fp, " %f %f ", &x, &y) != EOF)
    positions.emplace_back(x, y);
  std::cout << "particles read: " << positions.size() << std::endl;
  fclose(fp);
  return true;
}

void render() {
  glColor4f(0, 0, 0, 1);
  glBegin(GL_POINTS);
  for (auto p : positions)
    aergia::glVertex(p);
  glEnd();
  double elapsed = cl.tackTick();
  if (elapsed < 1000. / 30.)
#ifdef WIN32
    Sleep((1000. / 30. - elapsed) * 5);
#else
    sleep(1000. / 30. - elapsed);
#endif
  if (!readFrame(curFrame++))
    curFrame = 0;
}

int main(int argc, char **argv) {
  sscanf(argv[1], "%s", nameFormat);
  float dx;
  sscanf(argv[2], "%f", &dx);
  int nx, ny;
  sscanf(argv[3], "%d", &nx);
  sscanf(argv[4], "%d", &ny);
  readFrame(0);
  aergia::App app(800, 800, "View FLIP2D", false);
  app.addViewport2D(0, 0, 800, 800);
  app.viewports[0].renderCallback = render;
  static_cast<aergia::Camera2D *>(app.viewports[0].camera.get())
      ->setPos(ponos::vec2(nx, ny) * 0.5f * dx);
  static_cast<aergia::Camera2D *>(app.viewports[0].camera.get())
      ->setZoom(nx * dx);
  app.init();
  app.run();
  return 0;
}
