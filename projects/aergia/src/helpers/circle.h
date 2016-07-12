#pragma once

#include "utils/open_gl.h"

#include <ponos.h>
using ponos::vec2;

namespace aergia {

  class Circle {
  public:
    void draw() {
      glBegin(GL_TRIANGLE_FAN);
        glVertex(p);
        float angle = 0.0;
        float step = PI_2 / 10.f;
        while(angle < PI_2 + step) {
          vec2 pp(r * cosf(angle), r * sinf(angle));
          glVertex(p + pp);
          angle += step;
        }
      glEnd();
    }

    float r;
    vec2 p;
  };

}
