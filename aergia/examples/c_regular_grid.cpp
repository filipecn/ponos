#include <aergia/aergia.h>
#include <functional>
#include <ponos/ponos.h>
#include <queue>

using namespace aergia;

ponos::CRegularGrid2D<float> color[3];
ponos::RegularGrid2D<int> mask;
ponos::CRegularGrid2D<float> phi;

void copyBorder() {
  std::queue<ponos::ivec2> q;
  ponos::ivec2 ij;
  ponos::ivec2 dir[4] = {ponos::ivec2(-1, 0), ponos::ivec2(1, 0),
                         ponos::ivec2(0, -1), ponos::ivec2(0, 1)};
  FOR_INDICES0_2D(mask.getDimensions(), ij) {
    for (int i = 0; i < 4; i++) {
      if (ij[0] + dir[i][0] < 0 ||
          ij[0] + dir[i][0] >= mask.getDimensions()[0] ||
          ij[1] + dir[i][1] < 0 ||
          ij[1] + dir[i][1] >= mask.getDimensions()[1] || mask(ij) == 1)
        continue;
      q.push(ij);
      break;
    }
  }
  while (!q.empty()) {
    ponos::ivec2 p = q.front();
    q.pop();
    for (int i = 0; i < 4; i++) {
      if (mask(p + dir[i])) {
        phi(p) = phi(p + dir[i]);
        break;
      } else
        q.push(p + dir[i]);
    }
    mask(p) = 2;
  }
}

void render() {
  glPointSize(2.f);
  ponos::ivec2 ij;
  glBegin(GL_POINTS);
  FOR_INDICES0_2D(color[0].getDimensions(), ij) {
    if (mask(ij) != 1)
      continue;
    glColor(Color(color[0](ij), color[1](ij), color[2](ij)));
    glVertex(color[0].dataWorldPosition(ij));
  }
  // sample points
  ponos::ivec2 d(0);
  ponos::ivec2 D(100);
  FOR_INDICES2D(d, D, ij) {
    ponos::Point2 wp = color[0].dataWorldPosition(ij) / 10.f;
    // if (mask(ponos::ivec2(ij[0] / 10, ij[1] / 10)) == 1) {
    glColor(Color(color[0].sample(wp.x, wp.y), color[1].sample(wp.x, wp.y),
                  color[2].sample(wp.x, wp.y)));
    glVertex(wp);
    //}
  }
  glEnd();
}

int main() {
#ifdef WIN32
//  WIN32CONSOLE();
#endif
  mask.set(ponos::ivec2(10));
  mask.border = 0;
  ponos::ivec2 ij;
  FOR_INDICES0_2D(mask.getDimensions(), ij)
  if (ij[0] <= 3 || ij[0] >= 7 || ij[1] <= 4)
    mask(ij) = 1;
  else
    mask(ij) = 0;
  phi.set(10, 10, ponos::vec2(), ponos::vec2(1.f / 10.f));
  phi.accessMode = ponos::GridAccessMode::BORDER;
  phi.border = 1 << 16;
  std::vector<ponos::ivec2> points = {ponos::ivec2(0, 9)};
  ponos::fastMarch2D<ponos::CRegularGrid2D<float>, ponos::RegularGrid2D<int>,
                     int>(&phi, &mask, 1, points);
  copyBorder();
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++)
      std::cout << mask(ponos::ivec2(i, j)) << " ";
    std::cout << std::endl;
  }
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++)
      std::cout << phi(ponos::ivec2(j, i)) << "\t";
    std::cout << std::endl;
  }
  float maxDist = -1;
  FOR_INDICES0_2D(mask.getDimensions(), ij)
  if (mask(ij))
    maxDist = std::max(maxDist, phi(ij));
  std::cout << maxDist << std::endl;
  ColorPalette colorPalette = HEAT_MATLAB_PALETTE;
  for (int i = 0; i < 3; i++)
    color[i].set(10, 10, ponos::vec2(), ponos::vec2(1.f / 10.f));
  FOR_INDICES0_2D(color[0].getDimensions(), ij) {
    Color c = colorPalette(phi(ij) / maxDist);
    color[0](ij) = c.r;
    color[1](ij) = c.g;
    color[2](ij) = c.b;
  }
  SceneApp<> app(800, 400, "C Regular Grid Example", false);
  app.init();
  app.addViewport2D(0, 0, 800, 400);
  static_cast<Camera2D *>(app.viewports[0].camera.get())
      ->setPos(ponos::vec2(0.5));
  app.viewports[0].renderCallback = render;
  app.run();
  return 0;
}
