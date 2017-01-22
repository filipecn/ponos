#include <aergia.h>
#include <ponos.h>
#include <poseidon.h>
#include <iostream>
#include <vector>
#include <memory>

#include "flip_2d.h"
#include "flip_2d_model.h"

int WIDTH = 800, HEIGHT = 800;
int WGRID = 16, HGRID = 16;

aergia::SceneApp<> app(WIDTH, HEIGHT, "Z Grid example", false);

struct MySquare : public aergia::SceneObject {
	public:
		MySquare() {}
		MySquare(float r) {
			p.vertices.push_back(ponos::Point2(-1, -1));
			p.vertices.push_back(ponos::Point2(-1,  1));
			p.vertices.push_back(ponos::Point2( 1,  1));
			p.vertices.push_back(ponos::Point2( 1, -1));
			s = r;
			for(int i = 0; i < 4; i++)
				bbox_ = ponos::make_union(bbox_, p.vertices[i]);
		}
		void draw() const override {
			glLineWidth(3.f);
			glColor4f(0, 0, 1, 0.8);
			ponos::Transform2D t =
				ponos::translate(ponos::vec2(pos.x, pos.y)) *
				ponos::scale(s, s);
			aergia::draw_polygon(p, &t);
			glLineWidth(1.f);
		}
		ponos::BBox2D getBBox2D() {
			ponos::Transform2D t =
				ponos::translate(ponos::vec2(pos.x, pos.y)) *
				ponos::scale(s, s);
			return t(bbox_);
		}
		ponos::Polygon p;
		ponos::Point2 pos;
		ponos::BBox2D bbox_;
		float s;
};

MySquare sq;
ponos::HaltonSequence rng(3);

struct MyParticle : public aergia::SceneObject, public poseidon::FLIPParticle2D {
	public:
		MyParticle() {}
		MyParticle(const ponos::Point2 p) {
			this->position = p;
			c.c = p;
			c.r = 0.01f;
			color[0] = 0;
			color[1] = 0;
			color[2] = 1;
			color[3] = 0.1;
			this->velocity.x = ((rng.randomFloat() < 0.5f) ? 1 : -1) * rng.randomFloat();
			this->velocity.y = ((rng.randomFloat() < 0.5f) ? 1 : -1) * rng.randomFloat();
		}

		void draw() const override {
			glColor4fv(&color[0]);
			aergia::draw_circle(c);
		}
		bool intersect(const ponos::Ray3 &r, float *t = nullptr) override {
			return false;
			//return ponos::sphere_ray_intersection(s, r, t);
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
FLIP2D<MyParticle>* flip;
poseidon::ZParticleGrid2D<MyParticle>::tree* tree;

struct Tree : public aergia::SceneObject {
	public:
		void draw() const override {
			rec(tree->root);
		}

		void rec(poseidon::ZParticleGrid2D<MyParticle>::tree::Node *n) const {
			if(!n)
				return;
			float fcolor[4] = {0.1, 0.2, 0.5, 0.1};
			if(ponos::bbox_bbox_intersection(sq.getBBox2D(), flip->particleGrid->toWorld(n->bbox)))
				aergia::draw_bbox(flip->particleGrid->toWorld(n->bbox), &fcolor[0]);
			for(int i = 0; i < 4; i++)
				rec(n->child[i]);
		}
};

struct MacGridModel : public aergia::SceneObject {
	public:
		MacGridModel(MacGrid2D<ponos::ZGrid>* mg)
			: mgrid(mg) {}
		void draw() const override {
			aergia::CartesianGrid cgrid;
			cgrid.setDimension(0, 0, mgrid->dimensions[0]);
			cgrid.setDimension(1, 0, mgrid->dimensions[1]);
			cgrid.setDimension(2, 0, 0);
			cgrid.transform = mgrid->toWorld * ponos::translate(ponos::vec2(-0.5f, -0.5f));
			cgrid.draw();
			drawGridVelocities(mgrid->v_u.get(), 0);
			drawGridVelocities(mgrid->v_v.get(), 1);
		}

  void drawGridVelocities(ponos::ZGrid<float>* g, int component) const {
    glBegin(GL_LINES);
    for(uint32_t i = 0; i < g->width; ++i) {
      for(uint32_t j = 0; j < g->height; ++j) {
				ponos::Point2 wp = g->toWorld(ponos::Point2(i, j));
        aergia::glVertex(wp);
				ponos::vec2 v;
        v[component] = (*g)(i, j);
        aergia::glVertex(wp + v);
      }
    }
    glEnd();
  }
	MacGrid2D<ponos::ZGrid>* mgrid;
};

void search() {
	poseidon::ZParticleGrid2D<MyParticle>::particle_iterator it(*flip->particleGrid.get());
	while (it.next()) {
		(*it)->color[0] = 0;
		(*it)->color[3] = 0.3;
		++it;
	}
	tree->iterateParticles(sq.getBBox2D(), [](MyParticle* p) {
			p->color[0] = 1;
			p->color[3] = 1;

			});
}

void render() {
	poseidon::ZParticleGrid2D<MyParticle>::particle_iterator it(*flip->particleGrid.get());
	while (it.next()) {
		ponos::Point2 np = (*it)->position + 0.001f * (*it)->velocity;
		if(np.x < region.pMin.x) (*it)->velocity.x *= -1.f;
		if(np.y < region.pMin.y) (*it)->velocity.y *= -1.f;
		if(np.x > region.pMax.x) (*it)->velocity.x *= -1.f;
		if(np.y > region.pMax.y) (*it)->velocity.y *= -1.f;
		it.particleElement()->setPosition(*flip->particleGrid.get(), np);
		(*it)->c.c = (*it)->position;
		++it;
	}
	flip->particleGrid->update();
	if(tree)
		delete tree;
	tree = new poseidon::ZParticleGrid2D<MyParticle>::tree(*flip->particleGrid.get(),
			[](uint32 id, uint32 depth) { return true; }
			);
	search();
}

void mouse(double x, double y) {
	ponos::Point3 p = app.viewports[0].unProject();
	sq.pos = p.xy();
	search();
}

void init() {
	flip->dimensions = ponos::ivec2(WGRID, HGRID);
	flip->density = 1.f;
	flip->pic_flip_ratio = 0.98f;
	flip->subcell = 1;
	flip->dt = 0.001f;
	flip->dx = 0.1f;
	flip->curStep = 1;
	flip->setup();

	for (int i = 0; i < WGRID; i++)
		for (int j = 0; j < HGRID; j++)
			flip->particleGrid->add(ponos::Point2(i * flip->dx, j * flip->dx));
	flip->curGrid()->v_v->setAll(0.02f);
	flip->curGrid()->v_u->setAll(0.02f);
	poseidon::ZParticleGrid2D<MyParticle>::particle_iterator it(*flip->particleGrid.get());
	while (it.next()) {
		app.scene.add(*it);
		++it;
	}

	region = ponos::BBox2D(ponos::Point2(0, 0), ponos::Point2(WGRID, HGRID) * flip->dx);

	sq = MySquare(flip->dx * 1.5f);
	sq.pos = ponos::Point2(flip->dx, flip->dx) * 2.f;
	app.scene.add(&sq);
	app.scene.add(new Tree());
	app.scene.add(new MacGridModel(flip->curGrid()));
	app.scene.add(new FLIP2DSceneModel<MyParticle>(flip->scene.get()));
}

int main(int argc, char** argv) {
	flip = new FLIP2D<MyParticle>(new FLIP2DScene<MyParticle>(argv[1]));
	app.addViewport2D(0, 0, WIDTH, HEIGHT);
	app.init();
	init();
	static_cast<aergia::Camera2D*>(app.viewports[0].camera.get())->setPos(ponos::vec2(WGRID, HGRID) * 0.5f * flip->dx);
	static_cast<aergia::Camera2D*>(app.viewports[0].camera.get())->setZoom(region.pMax.x * 0.55f);
	app.viewports[0].mouseCallback = mouse;
	app.viewports[0].renderCallback = render;
	app.run();
	return 0;
}
