#include <aergia.h>
#include <ponos.h>
#include <poseidon.h>
#include <iostream>
#include <vector>
#include <memory>

#include "flip.h"

int WIDTH = 800, HEIGHT = 800;

aergia::SceneApp<> app(WIDTH, HEIGHT, "FLIP example");
poseidon::FLIP flip;

enum class GridObjectType {PARTICLE, U, V, W, P, T};

class GridObject : public aergia::SceneObject {
	public:
		GridObject(ponos::Point3 a, ponos::Point3 b, GridObjectType t, ponos::ivec3 coord)
			: type(t), ss(ponos::Segment3(a, b)) {
				data.coord[0] = coord[0];
				data.coord[1] = coord[1];
				data.coord[2] = coord[2];
				selected = false;
			}
		GridObject(ponos::Point3 p, float r, GridObjectType t, int id)
			: type(t), s(ponos::Sphere(p, r)) {
				selected = false;
				data.id = id; }
		GridObject(ponos::Point3 p, float r, GridObjectType t, ponos::ivec3 coord)
			: type(t), s(ponos::Sphere(p, r)) {
				data.coord[0] = coord[0];
				data.coord[1] = coord[1];
				data.coord[2] = coord[2];
				selected = false;
			}
		void draw() const override {
			switch(type) {
				case GridObjectType::PARTICLE: glColor4f(1,0,1,0.4); break;
				case GridObjectType::U: glColor4f(1,0,0,0.4); break;
				case GridObjectType::V: glColor4f(0,1,0,0.4); break;
				case GridObjectType::W: glColor4f(0,0,1,0.4); break;
				case GridObjectType::P: glColor4f(1,0,0,0.4); break;
				case GridObjectType::T: glColor4f(0,0,0,0.1);
								if(flip.cell(data.coord[0], data.coord[1], data.coord[2]) ==
										poseidon::FLIPCellType::FLUID)
																glColor4f(0,0,1,0.9);
								else if(flip.cell(data.coord[0], data.coord[1], data.coord[2]) ==
										poseidon::FLIPCellType::SOLID)
																glColor4f(1,0,0,0.9);
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
		union Data {
			int coord[3];
			int id;
		} data;
};

void buildGridObjects() {
	// add T
	ponos::ivec3 ijk;
	FOR_INDICES0_3D(flip.dimensions, ijk)
		app.scene.add(new GridObject(
					ponos::Point3(ijk[0], ijk[1], ijk[2]) * flip.dx, 0.05f,
					GridObjectType::T, ijk));
	// add U
	ponos::vec3 d(0.1f * flip.dx, 0.f, 0.f);
	FOR_INDICES0_3D((flip.dimensions + ponos::ivec3(1, 0, 0)), ijk) {
		ponos::Point3 p = ponos::Point3(ijk[0] - 0.5f, ijk[1], ijk[2]) * flip.dx;
		app.scene.add(new GridObject(p - d, p + d, GridObjectType::U, ijk));
	}
	// add V
	d = ponos::vec3(0.f, 0.1f * flip.dx, 0.f);
	FOR_INDICES0_3D((flip.dimensions + ponos::ivec3(0, 1, 0)), ijk) {
		ponos::Point3 p = ponos::Point3(ijk[0], ijk[1] - 0.5f, ijk[2]) * flip.dx;
		app.scene.add(new GridObject(p - d, p + d, GridObjectType::V, ijk));
	}
	// add W
	d = ponos::vec3(0.f, 0.f, 0.1f * flip.dx);
	FOR_INDICES0_3D((flip.dimensions + ponos::ivec3(0, 0, 1)), ijk) {
		ponos::Point3 p = ponos::Point3(ijk[0], ijk[1], ijk[2] - 0.5f) * flip.dx;
		app.scene.add(new GridObject(p - d, p + d, GridObjectType::W, ijk));
	}

	app.scene.add(new aergia::WireframeMesh(aergia::create_grid_mesh(
					flip.dimensions + ponos::ivec3(1),
					flip.dx, ponos::vec3(-flip.dx * 0.5f)), ponos::Transform()));
	const std::vector<aergia::Mesh*> &solids = flip.scene.getStaticSolidsGeometry();
	for(size_t i = 0; i < solids.size(); i++)
		app.scene.add(new aergia::WireframeMesh(aergia::create_wireframe_mesh(solids[i]->getMesh()), solids[i]->getTransform()));
	const std::vector<aergia::Mesh*> &liquids = flip.scene.getLiquidsGeometry();
	for(size_t i = 0; i < liquids.size(); i++)
		app.scene.add(new aergia::WireframeMesh(aergia::create_wireframe_mesh(liquids[i]->getMesh()), liquids[i]->getTransform()));
}

void init() {
	flip.scene.load("/home/filipecn/Desktop/flipscene");
	flip.dimensions = ponos::ivec3(10, 10, 10);
	flip.dx = 1.f;
	flip.setup();
	flip.density = 1.f;
	flip.init();
	buildGridObjects();
}

int main() {
	app.init();
	init();
	app.run();
	return 0;
}
