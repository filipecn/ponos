#ifndef circe_UI_INTERACTIVE_OBJECT_INTERFACE_H
#define circe_UI_INTERACTIVE_OBJECT_INTERFACE_H

#include <circe/ui/trackball_interface.h>

namespace circe {

class InteractiveObjectInterface {
public:
  InteractiveObjectInterface() : id(-1), selected(false), active(false) {}
  virtual ~InteractiveObjectInterface() {}

  virtual void mouse(CameraInterface &camera, ponos::point2 p) {
    trackball.mouseMove(camera, p);
    updateTransform();
  }

  virtual void button(CameraInterface &camera, ponos::point2 p, int button,
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

} // namespace circe

#endif // circe_UI_INTERACTIVE_OBJECT_INTERFACE_H
