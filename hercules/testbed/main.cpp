#include <aergia.h>
#include <ponos.h>

#include "gjk_demo.h"
#include "aabb_sweep_demo.h"
#include "gjk_transform_demo.h"


aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
Demo* curDemo;

void render() {
	curDemo->render();
}

int main() {
  std::shared_ptr<ponos::Polygon> sp = std::make_shared<ponos::Polygon>();
  std::shared_ptr<ponos::Polygon> ssp = std::make_shared<ponos::Polygon>();
  sp = ssp;
	//curDemo = new GJKDemo(800, 800);
	curDemo = new AABBSweepDemo(800, 800);
	//curDemo = new GJKTransformDemo(800, 800);
	curDemo->init();
  // init window
  aergia::createGraphicsDisplay(800, 800, "FLIP - 2D");
  gd.registerRenderFunc(render);
  gd.start();
  return 0;
}
