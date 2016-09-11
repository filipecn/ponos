#include <aergia.h>
#include <ponos.h>
#include <iostream>
#include <vector>
#include <memory>

int WIDTH = 800, HEIGHT = 800;

aergia::App app(WIDTH, HEIGHT, "FLIP example");
aergia::Scene<> scene;

class GridObject : public aergia::SceneObject {
	public:
		GridObject(ponos::CGridInterface<float>* g)
			: grid(g) {}
		virtual ~GridObject() {}

		void draw() const override {}

		bool intersect(const ponos::Ray3 &r, float *t = nullptr) const override {
			return false;
		}

	private:
		ponos::CGridInterface<float>* grid;
};

class ParticleObject : public aergia::SceneObject {
	public:
		ParticleObject(ponos::Point3 p, float r)
			: s(ponos::Sphere(p, r)), selected(false) {}
		void draw() const override {
			glColor4f(0,0,1,0.4);
			if(selected)
				glColor4f(1,0,0,0.4);
			aergia::draw_sphere(s);
		}
		bool intersect(const ponos::Ray3 &r, float *t = nullptr) const override {
			return ponos::sphere_ray_intersection(s, r, t);
		}
		ponos::Sphere s;
		bool selected;
};

std::vector<ParticleObject*> particles;
ParticleObject* selectedParticle;

void render() {
	scene.render();
}

void mouse(double x, double y) {
	ponos::Ray3 r = app.viewports[0].camera->pickRay(
			app.viewports[0].getMouseNPos());
	if(selectedParticle)
		selectedParticle->selected = false;
	selectedParticle = nullptr;
	selectedParticle = static_cast<ParticleObject*>(scene.intersect(r));
	if(selectedParticle)
		selectedParticle->selected = true;
}

int main() {
	scene.add(new aergia::CartesianGrid(5, 5, 5));
	// create particles
	for(int i = 0; i < 5; i++)
		for(int j = 0; j < 5; j++)
			for(int k = 0; k < 5; k++) {
				particles.push_back(new ParticleObject(
							ponos::Point3(i, j, k), 0.1f));
				scene.add(particles[particles.size() - 1]);
			}
	app.viewports[0].renderCallback = render;
	app.viewports[0].mouseCallback = mouse;
	app.run();
	return 0;
}
