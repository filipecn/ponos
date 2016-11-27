#include <aergia.h>
#include <ponos.h>

int WIDTH = 800, HEIGHT = 800;

aergia::SceneApp<> app(WIDTH, HEIGHT, "Stream Lines");
aergia::BVH* bvh;
ponos::CRegularGrid<ponos::vec3>* grid;

void init() {
  grid = new ponos::CRegularGrid<ponos::vec3>(ponos::ivec3(10, 10, 10), ponos::vec3());
  grid->setAll(ponos::vec3(0.3, 0, 0));
  bvh = new aergia::BVH(new aergia::SceneMesh("C:/Users/fuiri/Desktop/bunny.obj"));
  bvh->sceneMesh->transform = ponos::scale(50.f, 50.f, 50.f);
  app.scene.add(new aergia::WireframeMesh(aergia::create_wireframe_mesh(bvh->sceneMesh->rawMesh.get()),
    bvh->sceneMesh->transform));
  app.scene.add(new aergia::VectorGrid(*grid));
	app.scene.add(new aergia::BVHModel(bvh));
}

int main() {
  WIN32CONSOLE();
  app.init();
  init();
  app.run();
  return 0;
}
