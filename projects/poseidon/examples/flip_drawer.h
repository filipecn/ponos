#pragma once

#include <aergia.h>
#include <poseidon.h>
using namespace poseidon;

class FLIPDrawer {
public:
  FLIPDrawer(FLIP *f) {
    flip = f;
  }

  void drawParticles() {
    const std::vector<Particle2D> &particles = flip->particleGrid.getParticles();
    for (auto particle : particles) {
      aergia::Circle c;
      c.r = 0.1 * flip->dx;
      c.p.x = particle.p.x;
      c.p.y = particle.p.y;
      glColor4f(0.f,0.5f,0.2f,0.5f);
      c.draw();
    }
  }

  template<class T>
  void drawGrid(const ZGrid<T>& grid) {
    float hdx = flip->dx / 2.f;
    vec2 offset = grid.toWorld.getTranslate();
    glBegin(GL_LINES);
    for (int i = 0; i <= grid.width; ++i) {
      glVertex2f(static_cast<float>(i) * flip->dx + offset.x - hdx, offset.y - hdx);
      glVertex2f(static_cast<float>(i) * flip->dx + offset.x - hdx,
      offset.y + flip->dx * grid.height - hdx);
    }
    for (int i = 0; i <= grid.height; ++i) {
      glVertex2f(offset.x - hdx, static_cast<float>(i) * flip->dx + offset.y - hdx);
      glVertex2f(offset.x + flip->dx * grid.width - hdx,
      static_cast<float>(i) * flip->dx + offset.y - hdx);
    }
    glEnd();
  }

  void drawMACGrid() {
    glColor4f(1.f, 0.f, 0.f, 0.5f);
    //drawGrid<float>(flip->u);
    glColor4f(0.f, 0.f, 1.f, 0.5f);
    //drawGrid(flip->v);
    glColor4f(1.f, 0.f, 1.f, 0.5f);
    drawGrid(flip->p);

    glColor4f(1.f, 0.f, 0.f, 0.1f);

    Point2 wp = flip->u.toWorld(Point2(5, 5));
    Point2 wpmin = wp + vec2(-flip->dx, -flip->dx);
    Point2 wpmax = wp + vec2(flip->dx, flip->dx);
    glBegin(GL_QUADS);
    glVertex2f(wpmin.x, wpmin.y);
    glVertex2f(wpmax.x, wpmin.y);
    glVertex2f(wpmax.x, wpmax.y);
    glVertex2f(wpmin.x, wpmax.y);
    glEnd();

    glColor4f(0.f, 0.f, 0.f, 1.0f);
    glPointSize(3);
    glBegin(GL_POINTS);
    flip->particleGrid.iterateParticles(ponos::BBox2D(wpmin, wpmax),
    [](const Particle2D & p){
      aergia::glVertex(p.p);
    });
    glEnd();

  }

private:
  poseidon::FLIP *flip;
};
