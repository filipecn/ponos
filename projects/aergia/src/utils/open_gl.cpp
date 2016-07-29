#include "utils/open_gl.h"

namespace aergia {

  void glVertex(ponos::Point3 v) {
    glVertex3f(v.x, v.y, v.z);
  }

  void glVertex(ponos::Point2 v) {
    glVertex2f(v.x, v.y);
  }

  void glVertex(ponos::vec2 v) {
    glVertex2f(v.x, v.y);
  }

} // aergia namespace
