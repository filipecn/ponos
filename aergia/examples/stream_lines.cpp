#include <aergia.h>
#include <ponos.h>

int WIDTH = 800, HEIGHT = 800;

aergia::SceneApp<> app(WIDTH, HEIGHT, "Stream Lines");
aergia::BVH* bvh;
ponos::CRegularGrid<ponos::vec3>* grid;
ponos::LevelSet* levelSet;

class Sphere : public aergia::SceneObject {
	public:
		Sphere(ponos::Sphere b) {
				sphere.c = b.c;
				sphere.r = b.r;
				selected = false;
			}

		void draw() const override {
			glColor4f(0, 0, 0, 0.3);
			if(selected)
				glColor4f(1, 0, 0, 0.5);
			aergia::draw_sphere(sphere);
			glColor4f(0, 0, 0, 0.3);
		}

		bool intersect(const ponos::Ray3 &r, float *t = nullptr) override {
			selected = ponos::sphere_ray_intersection(sphere, r);
			return selected;
		}

		bool selected;
		ponos::Sphere sphere;
};

class Box : public aergia::SceneObject {
	public:
		Box(ponos::BBox b)
			: bbox(b) {
				selected = false;
			}

		void draw() const override {
			glLineWidth(1.f);
			if(selected)
				glLineWidth(4.f);
			aergia::draw_bbox(bbox);
			glLineWidth(1.f);

		}

		bool intersect(const ponos::Ray3 &r, float *t = nullptr) override {
			float h1, h2;
			selected = ponos::bbox_ray_intersection(bbox, r, h1, h2);
			return selected;
		}

		bool selected;
		ponos::BBox bbox;
};

class Line : public aergia::SceneObject {
	public:
		Line(ponos::Ray3 r) {
			ray = r;
		}

		void draw() const override {
			glBegin(GL_LINES);
				aergia::glVertex(ray.o);
				aergia::glVertex(ray.o + 1000.f * ray.d);
			glEnd();
		}

		ponos::Ray3 ray;
};

class BBoxSampler : public aergia::SceneObject {
	public:
		BBoxSampler(const ponos::BBox& b, uint size, aergia::BVH* _bvh)
			: bbox(b) {
				rng[0] = new ponos::HaltonSequence(2);
				rng[1] = new ponos::HaltonSequence(3);
				rng[2] = new ponos::HaltonSequence(5);
				ponos::vec3 d = bbox.pMax - bbox.pMin;
				while(size) {
					ponos::Point3 p(
							bbox.pMin.x + rng[0]->randomFloat() * d.x,
							bbox.pMin.y + rng[1]->randomFloat() * d.y,
							bbox.pMin.z + rng[2]->randomFloat() * d.z);
					if(bvh->isInside(p)) {
						points.emplace_back(p);
						size--;
					}
				}
			}

		void draw() const override {
			glColor4f(1, 0.1, 0.2, 0.8);
			glPointSize(4);
			glBegin(GL_POINTS);
				for(uint i = 0; i < points.size(); i++)
					aergia::glVertex(points[i]);
			glEnd();
			glPointSize(1);
		}

		std::vector<ponos::Point3> points;

	private:
		ponos::HaltonSequence* rng[3];
		ponos::BBox bbox;
};

class StreamLine : public aergia::SceneObject {
	public:
		StreamLine(aergia::BVH* b, ponos::CRegularGrid<ponos::vec3>* g, const ponos::Point3& o) {
			ponos::Point3 cur = o;
			points.emplace_back(cur);
			int i = 0;
			while(i++ < 100) {
				ponos::vec3 v = (*grid)(ponos::vec3(cur));
				cur = cur + 0.1f * v;
				if(!bvh->isInside(cur))
						break;
				points.emplace_back(cur);
			}
		}
		void draw() const override {
			glLineWidth(5);
			glColor4f(1, 0, 0.2, 1);
			glBegin(GL_LINES);
				for(uint i = 1; i < points.size(); i++) {
					aergia::glVertex(points[i - 1]);
					aergia::glVertex(points[i]);
				}
			glEnd();
			glLineWidth(1);
		}

	private:
		std::vector<ponos::Point3> points;
};

int main() {
	WIN32CONSOLE();
	app.init();
	bvh = new aergia::BVH(new aergia::SceneMesh("C:/Users/fuiri/Desktop/bunny.obj"));
	bvh->sceneMesh->transform = ponos::translate(
			ponos::vec3(7, 3, 5)) * ponos::scale(50.f, 50.f, 50.f);
	app.scene.add(new aergia::WireframeMesh(
				aergia::create_wireframe_mesh(bvh->sceneMesh->rawMesh.get()),
				bvh->sceneMesh->transform));
	app.scene.add(new aergia::BVHModel(bvh));
	grid = new ponos::CRegularGrid<ponos::vec3>(ponos::ivec3(15, 15, 15), ponos::vec3());
	ponos::ivec3 ijk;
	FOR_INDICES0_3D(grid->dimensions, ijk) {
	//	if(bvh->isInside(grid->worldPosition(ijk)))
		grid->set(ijk, 0.3f * ponos::vec3(
					std::min(1.f, 2.f * ijk[1] + 2.f * ijk[2]),
					std::min(1.f, 2.f * ijk[0] + 2.f * ijk[2]),
					std::min(1.f, 2.f * ijk[0] + 2.f * ijk[1])));
	}
	//Line l(ponos::Ray3(ponos::Point3(5,5,5), ponos::vec3(1.2, 1, 2.0)));
	//ponos::Transform inv = ponos::inverse(bvh->sceneMesh->transform);
	//std::cout << bvh->intersect(inv(l.ray), nullptr) << std::endl;
	//app.scene.add(&l);
	app.scene.add(new Box(ponos::BBox(ponos::Point3(0, 0, 0), ponos::Point3(2, 2, 2))));
	app.scene.add(new Sphere(ponos::Sphere(ponos::Point3(-3, 0, 0), 1.5f)));
	//app.scene.add(new aergia::VectorGrid(*grid));
	BBoxSampler samples(bvh->sceneMesh->getBBox(), 100, bvh);
	app.scene.add(&samples);
	std::vector<StreamLine> streams;
	for(uint i = 0; i < samples.points.size(); i++)
		streams.emplace_back(bvh, grid, samples.points[i]);
	//for(uint i = 0; i < streams.size(); i++)
	//	app.scene.add(&streams[i]);
	app.run();
	return 0;
}
