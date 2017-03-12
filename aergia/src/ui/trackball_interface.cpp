#include "ui/trackball_interface.h"

namespace aergia {

void TrackballInterface::draw() { modes[curMode]->draw(tb); }

void TrackballInterface::buttonRelease(CameraInterface &camera, int button,
                                       ponos::Point2 p) {
  modes[curMode]->stop(tb, static_cast<Camera &>(camera), p);
}

void TrackballInterface::buttonPress(const CameraInterface &camera, int button,
                                     ponos::Point2 p) {
  switch (button) {
  case GLFW_MOUSE_BUTTON_LEFT:
    curMode = 2;
    break;
  case GLFW_MOUSE_BUTTON_MIDDLE:
    curMode = 1;
    break;
  case GLFW_MOUSE_BUTTON_RIGHT:
    curMode = 0;
    break;
  }
  modes[curMode]->start(tb, camera, p);
}

void TrackballInterface::mouseMove(CameraInterface &camera, ponos::Point2 p) {
  modes[curMode]->update(tb, static_cast<Camera &>(camera), p);
}

void TrackballInterface::mouseScroll(CameraInterface &camera, ponos::vec2 d) {
  // modes[curMode]->update(tb, camera, d);
}

} // aergia namespace
