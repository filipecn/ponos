#include "utils/open_gl.h"

namespace aergia {

  void glVertex(Point2 v) {
    glVertex2f(v.x, v.y);
  }

  void glVertex(vec2 v) {
    glVertex2f(v.x, v.y);
  }

} // aergia namespace
