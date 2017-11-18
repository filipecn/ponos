#include <aergia/aergia.h>
#include <ponos/ponos.h>

int WIDTH = 800, HEIGHT = 800;

aergia::SceneApp<> app(WIDTH, HEIGHT, "Stream Lines");
aergia::BVH *bvh;
ponos::CRegularGrid<ponos::vec3> *grid;
ponos::LevelSet *levelSet;

class Sphere : public aergia::SceneObject {
public:
  Sphere(ponos::Sphere b) {
    sphere.c = b.c;
    sphere.r = b.r;
    selected = false;
  }

  void draw() const override {
    glColor4f(0, 0, 0, 0.3);
    if (selected)
      glColor4f(1, 0, 0, 0.5);
    aergia::draw_sphere(sphere);
    glColor4f(0, 0, 0, 0.3);
  }

  bool intersect(const ponos::Ray3 &r, float *t = nullptr) override {
    UNUSED_VARIABLE(t);
    selected = ponos::sphere_ray_intersection(sphere, r);
    return selected;
  }

  bool selected;
  ponos::Sphere sphere;
};

class Box : public aergia::SceneObject {
public:
  Box(ponos::BBox b) : bbox(b) { selected = false; }

  void draw() const override {
    glLineWidth(1.f);
    if (selected)
      glLineWidth(4.f);
    aergia::draw_bbox(bbox);
    glLineWidth(1.f);
  }

  bool intersect(const ponos::Ray3 &r, float *t = nullptr) override {
    UNUSED_VARIABLE(t);
    float h1, h2;
    selected = ponos::bbox_ray_intersection(bbox, r, h1, h2);
    return selected;
  }

  bool selected;
  ponos::BBox bbox;
};

class Line : public aergia::SceneObject {
public:
  Line(ponos::Ray3 r) { ray = r; }

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
  BBoxSampler() {}
  BBoxSampler(const ponos::BBox &b, uint size, aergia::BVH *_bvh) : bbox(b) {
    UNUSED_VARIABLE(_bvh);
    rng[0] = new ponos::HaltonSequence(2);
    rng[1] = new ponos::HaltonSequence(3);
    rng[2] = new ponos::HaltonSequence(5);
    ponos::vec3 d = bbox.pMax - bbox.pMin;
    while (size) {
      ponos::Point3 p(bbox.pMin.x + rng[0]->randomFloat() * d.x,
                      bbox.pMin.y + rng[1]->randomFloat() * d.y,
                      bbox.pMin.z + rng[2]->randomFloat() * d.z);
      if (bvh->isInside(p)) {
        points.emplace_back(p);
        size--;
      }
    }
  }

  void draw() const override {
    glColor4f(1, 0.1, 0.2, 0.8);
    glPointSize(4);
    glBegin(GL_POINTS);
    for (uint i = 0; i < points.size(); i++)
      aergia::glVertex(points[i]);
    glEnd();
    glPointSize(1);
  }

  std::vector<ponos::Point3> points;

private:
  ponos::HaltonSequence *rng[3];
  ponos::BBox bbox;
};

class FieldSampler : public BBoxSampler {
public:
  FieldSampler() {
    ponos::ivec3 ijk;
    FOR_INDICES0_3D(grid->dimensions, ijk)
    if (bvh->isInside(grid->worldPosition(ijk)))
      points.emplace_back(grid->worldPosition(ijk));
  }
};

class StreamLine : public aergia::SceneObject {
public:
  StreamLine(aergia::BVH *b, ponos::CRegularGrid<ponos::vec3> *g,
             const ponos::Point3 &o) {
    UNUSED_VARIABLE(g);
    UNUSED_VARIABLE(b);
    ponos::Point3 cur = o;
    points.emplace_back(cur);
    int i = 0;
    while (i++ < 100) {
      ponos::vec3 v = (*grid)(ponos::vec3(cur));
      cur = cur + 0.01f * v;
      if (!bvh->isInside(cur))
        break;
      points.emplace_back(cur);
    }
    i = 0;
    cur = o;
    while (i++ < 100) {
      ponos::vec3 v = (*grid)(ponos::vec3(cur));
      cur = cur - 0.01f * v;
      if (!bvh->isInside(cur))
        break;
      points.emplace(points.begin(), cur);
    }
    origin = o;
  }
  void draw() const override {
    glLineWidth(2);
    glColor4f(1, 0, 0.2, 0.3);
    glBegin(GL_LINES);
    for (uint i = 1; i < points.size(); i++) {
      aergia::glVertex(points[i - 1]);
      aergia::glVertex(points[i]);
    }

    //			glColor4f(1, 0, 0.8, 1);

    //				aergia::glVertex(origin);
    //				aergia::glVertex(origin +
    //(*grid)(ponos::vec3(origin)));

    glEnd();
    glLineWidth(1);
  }

  std::vector<ponos::Point3> points;

private:
  ponos::Point3 origin;
};

int main(int argc, char **argv) {
#if defined(WIN32) || defined(_WIN32)
  WIN32CONSOLE();
#endif
  if (argc < 3) {
    std::cout << "usage:\n <path to obj file> <n> <vector field file - n "
                 "stream lines : to generate VF>\n";
    return 0;
  }
  int n = 15;
  sscanf(argv[2], "%d", &n);
  app.init();
  bvh = new aergia::BVH(new aergia::SceneMesh(argv[1]));
  grid = new ponos::CRegularGrid<ponos::vec3>(ponos::ivec3(n), ponos::vec3(),
                                              bvh->sceneMesh->getBBox());
  app.scene.add(new aergia::WireframeMesh(
      aergia::create_wireframe_mesh(bvh->sceneMesh->rawMesh),
      bvh->sceneMesh->transform));
  app.scene.add(new Box(bvh->sceneMesh->getBBox()));
  ponos::ivec3 ijk;
  if (argc < 4) {
    FILE *fp = fopen("points", "w+");
    if (!fp)
      return 0;
    FOR_INDICES0_3D(grid->dimensions, ijk)
    if (bvh->isInside(grid->worldPosition(ijk))) {
      ponos::Point3 p = grid->worldPosition(ijk);
      fprintf(fp, "%f %f %f\n", p.x, p.y, p.z);
    }
    fclose(fp);
  }
  // read field
  FILE *fp = fopen(argv[3], "r");
  if (!fp)
    return 0;
  FOR_INDICES0_3D(grid->dimensions, ijk)
  if (bvh->isInside(grid->worldPosition(ijk))) {
    ponos::vec3 v;
    if (fscanf(fp, "%f %f %f", &v[0], &v[1], &v[2]))
      grid->set(ijk, v);
  }
  fclose(fp);
  grid->normalize();
  int nstreams = 1000;
  if (argc > 4)
    sscanf(argv[4], "%d", &nstreams);

  BBoxSampler samples(bvh->sceneMesh->getBBox(), nstreams, bvh);
  app.scene.add(&samples);
  std::vector<StreamLine> streams;
  for (uint i = 0; i < samples.points.size(); i++)
    streams.emplace_back(bvh, grid, samples.points[i]);
  for (uint i = 0; i < streams.size(); i++)
    app.scene.add(&streams[i]);
  app.run();
  char filename[100];
  sprintf(filename, "streams%d", nstreams);
  fp = fopen(filename, "w+");
  if (!fp)
    return 0;
  fprintf(fp, "%u\n", streams.size());
  for (uint i = 0; i < streams.size(); i++) {
    fprintf(fp, "%u ", streams[i].points.size());
    for (uint j = 0; j < streams[i].points.size(); j++)
      fprintf(fp, "%f %f %f ", streams[i].points[j].x, streams[i].points[j].y,
              streams[i].points[j].z);
    fprintf(fp, "\n");
  }
  fclose(fp);
  return 0;
}
