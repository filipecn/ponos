#pragma once

#include <aergia.h>
#include <ponos.h>
#include <poseidon.h>
using namespace poseidon;

class FLIPDrawer {
public:
  FLIPDrawer(FLIP *f) {
    flip = f;
  }

  void drawParticle(Particle2D particle) {
    aergia::Circle c;
    c.r = 0.1 * flip->dx;
    c.p.x = particle.p.x;
    c.p.y = particle.p.y;
    c.draw();
    glColor4f(0.f,0.5f,0.4f,1.0f);
    glBegin(GL_LINES);
    aergia::glVertex(c.p);
    aergia::glVertex(c.p + particle.v * flip->dt);
    glEnd();
  }

  void drawParticles() {
    const std::vector<Particle2D> &particles = flip->particleGrid.getParticles();
    for (auto particle : particles) {
      glColor4f(0.f,0.5f,0.2f,0.5f);
      drawParticle(particle);
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

  void drawGridVelocities(const ZGrid<FLIP::VelocityCell>& grid, int component) {
    glBegin(GL_LINES);
    for (int i = 0; i < grid.width; ++i) {
      for (int j = 0; j < grid.height; ++j) {
        Point2 wp = grid.toWorld(Point2(i, j));
        aergia::glVertex(wp);
        vec2 v;
        v[component] = grid(i, j).v;
        aergia::glVertex(wp + flip->dt * v);
      }
    }
    glEnd();
  }

  void drawMACGrid() {
    glColor4f(1.f, 0.f, 0.f, 0.2f);
    drawGrid(flip->u);
    glColor4f(0.f, 0.f, 1.f, 0.2f);
    drawGrid(flip->v);
    glColor4f(1.f, 0.f, 1.f, 0.5f);
    drawGrid(flip->cell);

    glColor4f(1.f, 0.f, 0.f, 0.1f);
/*
    Point2 wp = flip->v.toWorld(Point2(2, 2));
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
  */
    glPointSize(3);
    glBegin(GL_POINTS);
    Point2 wp = flip->particleGrid.getParticle(0).p;
    Point2 gp = flip->v.toGrid(flip->particleGrid.getParticle(0).p);
    glColor4f(1.f, 0.f, 0.f, 1.0f);
    aergia::glVertex(wp);
    glColor4f(0.f, 0.f, 0.f, 1.0f);
    int xmin = static_cast<int>(gp.x);
    int ymin = static_cast<int>(gp.y);
    for(int x = xmin; x <= xmin + 1; x++)
    for(int y = ymin; y <= ymin + 1; y++) {
      if(x < 0 || x >= flip->v.width || y < 0 || y >= flip->v.height)
      continue;
      Point2 gwp = flip->v.toWorld(Point2(x, y));
      aergia::glVertex(gwp);
    }
    glEnd();
  }

  void drawCells() {
    float hdx = flip->dx / 2.f;
    glBegin(GL_QUADS);
      for(int i = 0; i < flip->cell.width; i++)
      for (int j = 0; j < flip->cell.height; ++j)
      {
        glColor4f(1.f, 1.f, 1.f, 0.0f);
        if(flip->cell(i, j) == FLUID)
          glColor4f(0.f, 0.f, 1.f, 0.1f);
        else if(flip->cell(i, j) == SOLID)
          glColor4f(0.f, 0.f, 0.f, 0.1f);
        Point2 wp = flip->cell.toWorld(Point2(i, j));
        aergia::glVertex(wp + ponos::vec2(-hdx, -hdx));
        aergia::glVertex(wp + ponos::vec2( hdx, -hdx));
        aergia::glVertex(wp + ponos::vec2( hdx,  hdx));
        aergia::glVertex(wp + ponos::vec2(-hdx,  hdx));
      }
    glEnd();
  }

private:
  poseidon::FLIP *flip;
};
