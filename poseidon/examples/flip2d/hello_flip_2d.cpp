#include <aergia.h>
#include <ponos.h>
#include <poseidon.h>
#include <iostream>
#include <vector>
#include <memory>

#include "flip_2d.h"
#include "flip_2d_model.h"

int WIDTH = 800, HEIGHT = 800;

aergia::SceneApp<> app(WIDTH, HEIGHT, "Z Grid example", false);

struct MySquare : public aergia::SceneObject {
public:
  MySquare() {}
  MySquare(float r) {
    p.vertices.push_back(ponos::Point2(-1, -1));
    p.vertices.push_back(ponos::Point2(-1, 1));
    p.vertices.push_back(ponos::Point2(1, 1));
    p.vertices.push_back(ponos::Point2(1, -1));
    s = r;
    for (int i = 0; i < 4; i++)
      bbox_ = ponos::make_union(bbox_, p.vertices[i]);
  }
  void draw() const override {
    glLineWidth(3.f);
    glColor4f(0, 0, 1, 0.8);
    ponos::Transform2D t =
        ponos::translate(ponos::vec2(pos.x, pos.y)) * ponos::scale(s, s);
    aergia::draw_polygon(p, &t);
    glLineWidth(1.f);
  }
  ponos::BBox2D getBBox2D() {
    ponos::Transform2D t =
        ponos::translate(ponos::vec2(pos.x, pos.y)) * ponos::scale(s, s);
    return t(bbox_);
  }
  ponos::Polygon p;
  ponos::Point2 pos;
  ponos::BBox2D bbox_;
  float s;
};

MySquare sq;
ponos::HaltonSequence rng(3);

struct MyParticle : public aergia::SceneObject,
                    public poseidon::FLIPParticle2D {
public:
  MyParticle() {
    this->radius = 0.001f;
    c.r = 0.001f;
    color[0] = 0;
    color[1] = 0;
    color[2] = 1;
    color[3] = 0.1;
  }
  MyParticle(const ponos::Point2 p) {
    this->position = p;
    c.c = p;
    this->radius = 0.001f;
    c.r = 0.001f;
    color[0] = 0;
    color[1] = 0;
    color[2] = 1;
    color[3] = 0.1;
    this->velocity.x =
        ((rng.randomFloat() < 0.5f) ? 1 : -1) * rng.randomFloat();
    this->velocity.y =
        ((rng.randomFloat() < 0.5f) ? 1 : -1) * rng.randomFloat();
  }

  void draw() const override {
    glColor4fv(&color[0]);
    aergia::draw_circle(c);
  }
  bool intersect(const ponos::Ray3 &r, float *t = nullptr) override {
    return false;
    // return ponos::sphere_ray_intersection(s, r, t);
  }
  ponos::Circle c;
  ponos::Segment3 ss;
  union Data {
    int coord[3];
    int id;
  } data;
  float color[4];
};

ponos::BBox2D region;
FLIP2D<MyParticle> *flip;
poseidon::ZParticleGrid2D<MyParticle>::tree *tree;

struct Tree : public aergia::SceneObject {
public:
  void draw() const override { rec(tree->root); }

  void rec(poseidon::ZParticleGrid2D<MyParticle>::tree::Node *n) const {
    if (!n)
      return;
    float fcolor[4] = {0.1, 0.2, 0.5, 0.1};
    if (ponos::bbox_bbox_intersection(sq.getBBox2D(),
                                      flip->particleGrid->toWorld(n->bbox)))
      aergia::draw_bbox(flip->particleGrid->toWorld(n->bbox), &fcolor[0]);
    for (int i = 0; i < 4; i++)
      rec(n->child[i]);
  }
};

struct MacGridModel : public aergia::SceneObject {
public:
  MacGridModel(MacGrid2D<ponos::ZGrid> *mg) : mgrid(mg) {}
  void draw() const override {
    aergia::CartesianGrid cgrid;
    cgrid.setDimension(0, 0, mgrid->dimensions[0]);
    cgrid.setDimension(1, 0, mgrid->dimensions[1]);
    cgrid.setDimension(2, 0, 0);
    cgrid.transform =
        mgrid->toWorld * ponos::translate(ponos::vec2(-0.5f, -0.5f));
    cgrid.draw();
    drawGridVelocities(mgrid->v_u.get(), 0);
    drawGridVelocities(mgrid->v_v.get(), 1);
  }

  void drawGridVelocities(ponos::ZGrid<float> *g, int component) const {
    glColor4f(1.f, 0.f, 1.f, 0.3f);
    glBegin(GL_LINES);
    for (uint32_t i = 0; i < g->width; ++i) {
      for (uint32_t j = 0; j < g->height; ++j) {
        ponos::Point2 wp = g->toWorld(ponos::Point2(i, j));
        aergia::glVertex(wp);
        ponos::vec2 v;
        v[component] = (*g)(i, j);
        aergia::glVertex(wp + v);
      }
    }
    glEnd();
    // draw cell types
    ponos::ivec2 ij;
    FOR_INDICES0_2D(flip->dimensions, ij) {
      if ((*mgrid->cellType)(ij) != CellType::AIR) {
        glColor4f(0.f, 0.f, 1.f, 0.1f);
        if ((*mgrid->cellType)(ij) == CellType::SOLID)
          glColor4f(0.3f, 0.4f, 0.2f, 0.1f);
        aergia::fill_box(
            flip->particleGrid->toWorld(ponos::Point2(ij[0], ij[1])),
            flip->particleGrid->toWorld(ponos::Point2(ij[0] + 1, ij[1] + 1)));
      }
    }
  }
  MacGrid2D<ponos::ZGrid> *mgrid;
};

void search() {
  poseidon::ZParticleGrid2D<MyParticle>::particle_iterator it(
      *flip->particleGrid.get());
  while (it.next()) {
    (*it)->color[0] = 0;
    (*it)->color[3] = 0.3;
    ++it;
  }
  flip->particleGrid->update();
  flip->particleGrid->tree_->iterateParticles(sq.getBBox2D(),
                                              [](MyParticle *p) {
                                                p->color[0] = 1;
                                                p->color[3] = 1;
                                              });
}

void render() {
  // flip->step();
  poseidon::ZParticleGrid2D<MyParticle>::particle_iterator it(
      *flip->particleGrid.get());
  while (it.next()) {
    (*it)->c.c = (*it)->position;
    (*it)->draw();
    glColor4f(1.f, 0.f, 1.f, 0.5f);
    glBegin(GL_LINES);
    aergia::glVertex((*it)->c.c);
    aergia::glVertex((*it)->c.c + (*it)->velocity);
    glEnd();
    ++it;
  }
  search();
}

void mouse(double x, double y) {
  ponos::Point3 p = app.viewports[0].unProject();
  sq.pos = p.xy();
  search();
}

void button(int b, int a) {
  if (a == GLFW_RELEASE) {
    flip->step();
    poseidon::ZParticleGrid2D<MyParticle>::particle_iterator it(
        *flip->particleGrid.get());
    std::cout << it.count() << std::endl;
    std::cout << flip->particleGrid->elementCount() << std::endl
              << std::endl;
    return;
    while (it.next()) {
      if ((*it)->type == poseidon::ParticleTypes::FLUID) {
        std::cout << (*it)->velocity << std::endl;
        break;
      }
      ++it;
    }
  }
}

void init() {
  flip->init();
  flip->scene->addForce(ponos::vec2(0.f, -9.8f));

  // flip->getGrid()->v_v->setAll(0.02f);
  // fli>getGrid()->v_u->setAll(0.02f);

  region = ponos::BBox2D(
      ponos::Point2(0, 0),
      ponos::Point2(flip->dimensions[0], flip->dimensions[1]) * flip->dx);

  sq = MySquare(flip->dx * 1.5f);
  sq.pos = ponos::Point2(flip->dx, flip->dx) * 2.f;
  app.scene.add(&sq);
  // app.scene.add(new Tree());
  app.scene.add(new MacGridModel(flip->getGrid()));
  app.scene.add(new FLIP2DSceneModel<MyParticle>(flip));
}

int main(int argc, char **argv) {
  flip = new FLIP2D<MyParticle>(argv[1]);
  flip->loadScene(argv[2]);
  app.addViewport2D(0, 0, WIDTH, HEIGHT);
  app.init();
  init();
  static_cast<aergia::Camera2D *>(app.viewports[0].camera.get())
      ->setPos(ponos::vec2(flip->dimensions[0], flip->dimensions[1]) * 0.5f *
               flip->dx);
  static_cast<aergia::Camera2D *>(app.viewports[0].camera.get())
      ->setZoom(region.size(0) * 2.f);
  app.viewports[0].mouseCallback = mouse;
  app.viewports[0].renderCallback = render;
  app.viewports[0].buttonCallback = button;
  app.run();
  return 0;
}
