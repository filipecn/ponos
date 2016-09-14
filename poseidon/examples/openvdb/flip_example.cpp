#include <aergia.h>
#include <ponos.h>
#include <poseidon.h>
#include <iostream>
#include <vector>
#include <memory>

#include "flip.h"

int WIDTH = 800, HEIGHT = 800;

aergia::App app(WIDTH, HEIGHT, "FLIP example");
aergia::Scene<> scene;
poseidon::FLIP flip;

enum class GridObjectType {PARTICLE, U, V, W, P, T};

class Grid : public aergia::SceneObject {
	public:
		Grid(ponos::ivec3 pmin, ponos::ivec3 pmax, float s)
			: pMin(pmin), pMax(pmax), scale(s) {}
		void draw() const override {
			glColor4f(0,0,0,0.05f);
			glBegin(GL_LINES);
			ponos::ivec2 ij;
			ponos::ivec2 p0 = pMin.xy();
			ponos::ivec2 p1 = pMax.xy() + ponos::ivec2(1, 1);
			FOR_INDICES2D(p0, p1, ij) {
				glVertex3f((ij[0] - 0.5f) * scale, (ij[1] - 0.5f) * scale, (pMin[2] - 0.5f) * scale);
				glVertex3f((ij[0] - 0.5f) * scale, (ij[1] - 0.5f) * scale, (pMax[2] - 0.5f) * scale);
			}
			p0 = pMin.xy(0, 2);
			p1 = pMax.xy(0, 2) + ponos::ivec2(1, 1);
			FOR_INDICES2D(p0, p1, ij) {
				glVertex3f((ij[0] - 0.5f) * scale, (pMin[1] - 0.5f) * scale, (ij[1] - 0.5f) * scale);
				glVertex3f((ij[0] - 0.5f) * scale, (pMax[1] - 0.5f) * scale, (ij[1] - 0.5f) * scale);
			}
			p0 = pMin.xy(1, 2);
			p1 = pMax.xy(1, 2) + ponos::ivec2(1, 1);
			FOR_INDICES2D(p0, p1, ij) {
				glVertex3f((pMin[0] - 0.5f) * scale, (ij[0] - 0.5f) * scale, (ij[1] - 0.5f) * scale);
				glVertex3f((pMax[0] - 0.5f) * scale, (ij[0] - 0.5f) * scale, (ij[1] - 0.5f) * scale);
			}
			glEnd();
		}
		ponos::ivec3 pMin, pMax;
		float scale;
};

class GridObject : public aergia::SceneObject {
	public:
		GridObject(ponos::Point3 a, ponos::Point3 b, GridObjectType t, ponos::ivec3 coord)
			: type(t), ss(ponos::Segment3(a, b)), selected(false) {
				data.coord[0] = coord[0];
				data.coord[1] = coord[1];
				data.coord[2] = coord[2];
			}
		GridObject(ponos::Point3 p, float r, GridObjectType t, int id)
			: type(t), s(ponos::Sphere(p, r)), selected(false) { data.id = id; }
		GridObject(ponos::Point3 p, float r, GridObjectType t, ponos::ivec3 coord)
			: type(t), s(ponos::Sphere(p, r)), selected(false) {
				data.coord[0] = coord[0];
				data.coord[1] = coord[1];
				data.coord[2] = coord[2];
			}
		void draw() const override {
			switch(type) {
				case GridObjectType::PARTICLE: glColor4f(1,0,1,0.4); break;
				case GridObjectType::U: glColor4f(1,0,0,0.4); break;
				case GridObjectType::V: glColor4f(0,1,0,0.4); break;
				case GridObjectType::W: glColor4f(0,0,1,0.4); break;
				case GridObjectType::P: glColor4f(1,0,0,0.4); break;
				case GridObjectType::T: glColor4f(0,0,0,0.1);
								if(flip.cell[data.coord[0]][data.coord[1]][data.coord[2]] ==
										poseidon::FLIPCellType::FLUID)
																glColor4f(0,0,1,0.9);
								break;
			}
			glLineWidth(1.0f);
			if(selected) {
				glColor4f(1,0,0,1);
				glLineWidth(6.0f);
			}
			switch(type) {
				case GridObjectType::U:
				case GridObjectType::V:
				case GridObjectType::W:
					aergia::draw_segment(ss); break;
				default:
					aergia::draw_sphere(s);
			}
		}
		bool intersect(const ponos::Ray3 &r, float *t = nullptr) const override {
			if(type == GridObjectType::U ||
					type == GridObjectType::V ||
					type == GridObjectType::W)
				return ponos::ray_segment_intersection(r, ss, t);
			return ponos::sphere_ray_intersection(s, r, t);
		}
		GridObjectType type;
		ponos::Sphere s;
		ponos::Segment3 ss;
		bool selected;
		union Data {
			int coord[3];
			int id;
		} data;
};

GridObject* selectedObject;

void init() {
	flip.dimensions = ponos::ivec3(10, 10, 10);
	flip.density = 1.f;
	flip.dx = 1.f;
	flip.setup();
	int k = 0;
	ponos::Point3 xyz;
	FOR_INDICES(4, 7, 4, 7, 4, 7, xyz) {
		flip.particleGrid->addParticle(xyz, ponos::vec3());
		scene.add(new GridObject(xyz, 0.1f, GridObjectType::PARTICLE, k++));
	}
	// add T
	ponos::ivec3 ijk;
	FOR_INDICES0_3D(flip.dimensions, ijk)
		scene.add(new GridObject(
					ponos::Point3(ijk[0], ijk[1], ijk[2]) * flip.dx, 0.05f,
					GridObjectType::T, ijk));
	// add U
	ponos::vec3 d(0.1f * flip.dx, 0.f, 0.f);
	FOR_INDICES0_3D((flip.dimensions + ponos::ivec3(1, 0, 0)), ijk) {
		ponos::Point3 p = ponos::Point3(ijk[0] - 0.5f, ijk[1], ijk[2]) * flip.dx;
		scene.add(new GridObject(p - d, p + d, GridObjectType::U, ijk));
	}
	// add V
	d = ponos::vec3(0.f, 0.1f * flip.dx, 0.f);
	FOR_INDICES0_3D((flip.dimensions + ponos::ivec3(0, 1, 0)), ijk) {
		ponos::Point3 p = ponos::Point3(ijk[0], ijk[1] - 0.5f, ijk[2]) * flip.dx;
		scene.add(new GridObject(p - d, p + d, GridObjectType::V, ijk));
	}
	// add W
	d = ponos::vec3(0.f, 0.f, 0.1f * flip.dx);
	FOR_INDICES0_3D((flip.dimensions + ponos::ivec3(0, 0, 1)), ijk) {
		ponos::Point3 p = ponos::Point3(ijk[0], ijk[1], ijk[2] - 0.5f) * flip.dx;
		scene.add(new GridObject(p - d, p + d, GridObjectType::W, ijk));
	}
	scene.add(new Grid(ponos::ivec3(), ponos::ivec3(10, 10, 10), flip.dx));
	flip.init();
}

void render() {
	scene.render();
}

void mouse(double x, double y) {
	ponos::Ray3 r = app.viewports[0].camera->pickRay(
			app.viewports[0].getMouseNPos());
	if(selectedObject)
		selectedObject->selected = false;
	selectedObject = static_cast<GridObject*>(scene.intersect(r));
	if(selectedObject)
		selectedObject->selected = true;
	scene.transform = app.trackball.tb.transform * scene.transform;
}

int main() {
	init();
	//scene.add(new aergia::CartesianGrid(5, 5, 5));
	app.viewports[0].renderCallback = render;
	app.viewports[0].mouseCallback = mouse;
	app.run();
	return 0;
}
