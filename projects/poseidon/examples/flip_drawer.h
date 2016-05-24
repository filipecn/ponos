#pragma once

#include <aergia.h>
#include <poseidon.h>
using namespace poseidon;

class FLIPDrawer {
public:
  FLIPDrawer(FLIP *f) {
    flip = f;
  }

  void drawGrid() {
    glColor3f(1.f,1.f,1.f);
    float hdx = flip->dx / 2.f;
    vec2 offset = flip->particleGrid.grid.toWorld.getTranslate();
    glBegin(GL_LINES);
    for (int i = 0; i < flip->particleGrid.grid.width; ++i) {
      glVertex2f(static_cast<float>(i) * flip->dx + offset.x, offset.y - hdx);
      glVertex2f(static_cast<float>(i) * flip->dx + offset.x, offset.y + hdx);
    }
    for (int i = 0; i < flip->particleGrid.grid.height; ++i) {
      glVertex2f(offset.x - hdx, static_cast<float>(i) * flip->dx + offset.y);
      glVertex2f(offset.y + hdx, static_cast<float>(i) * flip->dx + offset.y);
    }
    glEnd();
  }

private:
  poseidon::FLIP *flip;
};
