#include <aergia.h>
#include <ponos.h>
#include <poseidon.h>
#include <iostream>
#include <vector>
#include <memory>

int WIDTH = 800, HEIGHT = 800;

aergia::SceneApp<> app(WIDTH, HEIGHT, "Z Grid example", false);
aergia::CartesianGrid* cartesianGrid;

struct MySquare : public aergia::SceneObject {
public:
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

MySquare sq(1.5);

struct MyParticle : public aergia::SceneObject, public poseidon::FLIPParticle2D {
	public:
		MyParticle() {}
		MyParticle(const ponos::Point2 p) {
			this->position = p;
			s.c = ponos::Point3(p.x, p.y, 0.f);
			s.r = 0.1f;
			c.c = p;
			c.r = 0.1f;
			color[0] = 0;
			color[1] = 0;
			color[2] = 1;
			color[3] = 0.1;
		}

		void draw() const override {
			glColor4fv(&color[0]);
			aergia::draw_circle(c);
		}
		bool intersect(const ponos::Ray3 &r, float *t = nullptr) const override {
			return false;
			//return ponos::sphere_ray_intersection(s, r, t);
		}
		ponos::Sphere s;
		ponos::Circle c;
		ponos::Segment3 ss;
		union Data {
			int coord[3];
			int id;
		} data;
		float color[4];
};

poseidon::ZParticleGrid2D<MyParticle> grid(16, 16, ponos::scale(1, 1));
poseidon::ZParticleGrid2D<MyParticle>::tree* tree;

void mouse(double x, double y) {
	ponos::Point3 p = app.viewports[0].unProject();
	sq.pos = p.xy();
	tree->iterateParticles(sq.getBBox2D(), [](MyParticle* p) {
	  p->color[1] = 1;
	});
}

void init() {
	for (int i = 0; i < 16; i++)
		for (int j = 0; j < 16; j++)
			grid.add(ponos::Point2(i, j));
	grid.update();
	tree = new poseidon::ZParticleGrid2D<MyParticle>::tree(grid,
			[](uint32 id, uint32 depth) { return true; }
			);
	tree->iterateParticles(sq.getBBox2D(), [](MyParticle* p) {
	  p->color[1] = 1;
	});
	poseidon::ZParticleGrid2D<MyParticle>::particle_iterator it(grid);
	while (it.next()) {
		app.scene.add(*it);
		++it;
	}
	cartesianGrid = new aergia::CartesianGrid(16);
	cartesianGrid->transform = ponos::translate(ponos::vec3(-0.5f, -0.5f, 0.f));
	cartesianGrid->setDimension(0, 0, 16);
	cartesianGrid->setDimension(1, 0, 16);
	app.scene.add(cartesianGrid);
	sq.pos = ponos::Point2(8, 8);
	app.scene.add(&sq);
}

int main() {
	app.addViewport2D(0, 0, WIDTH, HEIGHT);
	app.init();
	static_cast<aergia::Camera2D*>(app.viewports[0].camera.get())->setPos(ponos::vec2(7.5, 7.5));
	static_cast<aergia::Camera2D*>(app.viewports[0].camera.get())->setZoom(9);
	app.viewports[0].mouseCallback = mouse;
	init();
	app.run();
	return 0;
}
