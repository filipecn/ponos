#ifndef AERGIA_UI_INTERACTIVE_OBJECT_INTERFACE_H
#define AERGIA_UI_INTERACTIVE_OBJECT_INTERFACE_H

#include <aergia/ui/trackball_interface.h>

namespace aergia {

class InteractiveObjectInterface {
public:
  InteractiveObjectInterface() : id(-1), selected(false), active(false) {}
  virtual ~InteractiveObjectInterface() {}

  virtual void mouse(CameraInterface &camera, ponos::Point2 p) {
    trackball.mouseMove(camera, p);
    updateTransform();
  }

  virtual void button(CameraInterface &camera, ponos::Point2 p, int button,
                      int action) {
    if (action == GLFW_RELEASE) {
      trackball.buttonRelease(camera, button, p);
      active = false;
    } else {
      trackball.buttonPress(camera, button, p);
      active = true;
    }
    updateTransform();
  }

  virtual void updateTransform() = 0;

  int id;
  bool selected;
  bool active;
  TrackballInterface trackball;
};

} // aergia namespace

#endif // AERGIA_UI_INTERACTIVE_OBJECT_INTERFACE_H
