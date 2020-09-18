#include <ponos.h>
#include <poseidon.h>
#include <cppunit/extensions/HelperMacros.h>

using namespace poseidon;

class TestZParticleGrid : public CppUnit::TestCase {
public:
  CPPUNIT_TEST_SUITE(TestZParticleGrid);
  CPPUNIT_TEST(testParticleIterator);
  CPPUNIT_TEST(testUpdate);
  CPPUNIT_TEST(testTree);
  CPPUNIT_TEST(testRemove);
  CPPUNIT_TEST(testGather);
  CPPUNIT_TEST_SUITE_END();

  void testParticleIterator();
  void testUpdate();
  void testTree();
  void testRemove();
  void testGather();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestZParticleGrid);

void TestZParticleGrid::testParticleIterator() {
  ZParticleGrid2D<FLIPParticle2D> grid(16, 16, ponos::scale(1, 1));
  std::vector<FLIPParticle2D> v;
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++)
      v.emplace_back(ponos::Point2(i, j));

  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++)
      grid.add(ponos::Point2(i, j));

  size_t i = 0;
  for (ZParticleGrid2D<FLIPParticle2D>::particle_iterator it(grid); it.next();
       ++it) {
    auto p = *it;
    CPPUNIT_ASSERT(v[i].position.x == p->position.x);
    CPPUNIT_ASSERT(v[i].position.y == p->position.y);
    i++;
  }
  CPPUNIT_ASSERT(i == 16 * 16);
}

void TestZParticleGrid::testUpdate() {
  ZParticleGrid2D<FLIPParticle2D> grid(8, 8, ponos::scale(1, 1));
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
      grid.add(ponos::Point2(i, j));
  grid.update();
  std::vector<ponos::Point2> order;
  int ddx = 0, ddy = 0;
  int dx = 0 + ddx, dy = 0 + ddy;
  order.emplace_back(0 + dx, 0 + dy);
  order.emplace_back(1 + dx, 0 + dy);
  order.emplace_back(0 + dx, 1 + dy);
  order.emplace_back(1 + dx, 1 + dy);
  dx = 2 + ddx;
  dy = 0 + ddy;
  order.emplace_back(0 + dx, 0 + dy);
  order.emplace_back(1 + dx, 0 + dy);
  order.emplace_back(0 + dx, 1 + dy);
  order.emplace_back(1 + dx, 1 + dy);
  dx = 0 + ddx;
  dy = 2 + ddy;
  order.emplace_back(0 + dx, 0 + dy);
  order.emplace_back(1 + dx, 0 + dy);
  order.emplace_back(0 + dx, 1 + dy);
  order.emplace_back(1 + dx, 1 + dy);
  dx = 2 + ddx;
  dy = 2 + ddy;
  order.emplace_back(0 + dx, 0 + dy);
  order.emplace_back(1 + dx, 0 + dy);
  order.emplace_back(0 + dx, 1 + dy);
  order.emplace_back(1 + dx, 1 + dy);

  ddx = 4;
  ddy = 0;
  dx = 0 + ddx;
  dy = 0 + ddy;
  order.emplace_back(0 + dx, 0 + dy);
  order.emplace_back(1 + dx, 0 + dy);
  order.emplace_back(0 + dx, 1 + dy);
  order.emplace_back(1 + dx, 1 + dy);
  dx = 2 + ddx;
  dy = 0 + ddy;
  order.emplace_back(0 + dx, 0 + dy);
  order.emplace_back(1 + dx, 0 + dy);
  order.emplace_back(0 + dx, 1 + dy);
  order.emplace_back(1 + dx, 1 + dy);
  dx = 0 + ddx;
  dy = 2 + ddy;
  order.emplace_back(0 + dx, 0 + dy);
  order.emplace_back(1 + dx, 0 + dy);
  order.emplace_back(0 + dx, 1 + dy);
  order.emplace_back(1 + dx, 1 + dy);
  dx = 2 + ddx;
  dy = 2 + ddy;
  order.emplace_back(0 + dx, 0 + dy);
  order.emplace_back(1 + dx, 0 + dy);
  order.emplace_back(0 + dx, 1 + dy);
  order.emplace_back(1 + dx, 1 + dy);

  ddx = 0;
  ddy = 4;
  dx = 0 + ddx;
  dy = 0 + ddy;
  order.emplace_back(0 + dx, 0 + dy);
  order.emplace_back(1 + dx, 0 + dy);
  order.emplace_back(0 + dx, 1 + dy);
  order.emplace_back(1 + dx, 1 + dy);
  dx = 2 + ddx;
  dy = 0 + ddy;
  order.emplace_back(0 + dx, 0 + dy);
  order.emplace_back(1 + dx, 0 + dy);
  order.emplace_back(0 + dx, 1 + dy);
  order.emplace_back(1 + dx, 1 + dy);
  dx = 0 + ddx;
  dy = 2 + ddy;
  order.emplace_back(0 + dx, 0 + dy);
  order.emplace_back(1 + dx, 0 + dy);
  order.emplace_back(0 + dx, 1 + dy);
  order.emplace_back(1 + dx, 1 + dy);
  dx = 2 + ddx;
  dy = 2 + ddy;
  order.emplace_back(0 + dx, 0 + dy);
  order.emplace_back(1 + dx, 0 + dy);
  order.emplace_back(0 + dx, 1 + dy);
  order.emplace_back(1 + dx, 1 + dy);

  ddx = 4;
  ddy = 4;
  dx = 0 + ddx;
  dy = 0 + ddy;
  order.emplace_back(0 + dx, 0 + dy);
  order.emplace_back(1 + dx, 0 + dy);
  order.emplace_back(0 + dx, 1 + dy);
  order.emplace_back(1 + dx, 1 + dy);
  dx = 2 + ddx;
  dy = 0 + ddy;
  order.emplace_back(0 + dx, 0 + dy);
  order.emplace_back(1 + dx, 0 + dy);
  order.emplace_back(0 + dx, 1 + dy);
  order.emplace_back(1 + dx, 1 + dy);
  dx = 0 + ddx;
  dy = 2 + ddy;
  order.emplace_back(0 + dx, 0 + dy);
  order.emplace_back(1 + dx, 0 + dy);
  order.emplace_back(0 + dx, 1 + dy);
  order.emplace_back(1 + dx, 1 + dy);
  dx = 2 + ddx;
  dy = 2 + ddy;
  order.emplace_back(0 + dx, 0 + dy);
  order.emplace_back(1 + dx, 0 + dy);
  order.emplace_back(0 + dx, 1 + dy);
  order.emplace_back(1 + dx, 1 + dy);

  CPPUNIT_ASSERT(order.size() == 8 * 8);
  int i = 0;
  for (ZParticleGrid2D<FLIPParticle2D>::particle_iterator it(grid); it.next();
       ++it) {
    auto p = *it;
    CPPUNIT_ASSERT(order[i].x == p->position.x);
    CPPUNIT_ASSERT(order[i].y == p->position.y);
    i++;
  }
}

void TestZParticleGrid::testTree() {
  ZParticleGrid2D<FLIPParticle2D> grid(8, 8, ponos::scale(1, 1));
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
      grid.add(ponos::Point2(i, j));
  grid.update();
  ZParticleGrid2D<FLIPParticle2D>::tree tree(
      grid, [&grid](uint32 id, uint32 depth) {
        ZParticleGrid2D<FLIPParticle2D>::particle_iterator it(grid, id, depth);
        // std::cout << it.count() << std::endl;
        return true;
      });
}

void TestZParticleGrid::testRemove() {
  ZParticleGrid2D<FLIPParticle2D> grid(8, 8, ponos::scale(1, 1));
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
      grid.add(ponos::Point2(i + .5f, j + .5f));
  CPPUNIT_ASSERT(grid.elementCount() == 64);
  grid.update();
  for (ZParticleGrid2D<FLIPParticle2D>::particle_iterator it(grid); it.next();
       ++it) {
    CPPUNIT_ASSERT(it.count() == 64);
    CPPUNIT_ASSERT(it.particleElement()->active);
  }
  ponos::ivec2 ij;
  FOR_INDICES0_2D(grid.dimensions, ij) {
    typename ZParticleGrid2D<FLIPParticle2D>::particle_iterator it =
        grid.getCell(ij);
    CPPUNIT_ASSERT(it.count() == 1);
  }
  ponos::ivec2 b(4, 4);
  int removeCount = 0;
  FOR_INDICES2D(b, grid.dimensions, ij) {
    for (typename ZParticleGrid2D<FLIPParticle2D>::particle_iterator it =
             grid.getCell(ij);
         it.next(); ++it) {
      it.particleElement()->active = false;
      removeCount++;
    }
  }
  grid.update();
  for (ZParticleGrid2D<FLIPParticle2D>::particle_iterator it(grid); it.next();
       ++it) {
    CPPUNIT_ASSERT(it.count() == 48);
    CPPUNIT_ASSERT(it.particleElement()->active);
  }
  CPPUNIT_ASSERT(grid.elementCount() == 64);
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
      grid.add(ponos::Point2(i + .5f, j + .5f));
  grid.update();
  for (ZParticleGrid2D<FLIPParticle2D>::particle_iterator it(grid); it.next();
       ++it) {
    CPPUNIT_ASSERT(it.count() == 64 + 48);
    CPPUNIT_ASSERT(it.particleElement()->active);
  }
  CPPUNIT_ASSERT(grid.elementCount() == 64 + 48);
}

void TestZParticleGrid::testGather() {
  ZParticleGrid2D<FLIPParticle2D> grid(8, 8, ponos::scale(1, 1));
  FLIPParticle2D *p = grid.add(ponos::Point2(0, 0));
  p->velocity = ponos::vec2(0.01, 10);
  p = grid.add(ponos::Point2(1, 0));
  p->velocity = ponos::vec2(0.01, 20);
  p = grid.add(ponos::Point2(1, 1));
  p->velocity = ponos::vec2(0.01, 30);
  p = grid.add(ponos::Point2(0, 1));
  p->velocity = ponos::vec2(0.01, 40);
  grid.update();
  /*
std::cout << "PARTICULAS\n";
for (ZParticleGrid2D<FLIPParticle2D>::particle_iterator it(grid); it.next();
 ++it) {
std::cout << (*it)->position;
}
std::cout << "BUSCA\n";
ponos::BBox2D b(c - ponos::vec2(2.f), c + ponos::vec2(2.f));
ZParticleGrid2D<FLIPParticle2D>::tree tree(
grid, [&grid](uint32 id, uint32 depth) { return true; });
tree.iterateParticles(b, [](FLIPParticle2D *o) { std::cout << o->position; });
*/
  ponos::Point2 c(0.5f, 0.5f);
  {
    float vx = grid.gather(ParticleAttribute::VELOCITY_X, c, 2);
    float vy = grid.gather(ParticleAttribute::VELOCITY_Y, c, 2);
    std::cout << vx << " " << vy << std::endl;
    CPPUNIT_ASSERT(IS_EQUAL(vx, 0.01) && IS_EQUAL(vy, 25.f));
  }
  p = grid.add(c);
  p->velocity = ponos::vec2(0.01, 100);
  grid.update();
  {
    float vx = grid.gather(ParticleAttribute::VELOCITY_X, c, 0.51);
    float vy = grid.gather(ParticleAttribute::VELOCITY_Y, c, 0.51);
    std::cout << vx << " " << vy << std::endl;
    // CPPUNIT_ASSERT(IS_EQUAL(vx, 0.01) && IS_EQUAL(vy, 25.f));
  }
  c = ponos::Point2(5.5, 5.5);
  p = grid.add(ponos::Point2(5.25, 5.25));
  p->velocity = ponos::vec2(0.01, 10);
  p = grid.add(ponos::Point2(5.75, 5.25));
  p->velocity = ponos::vec2(0.01, 20);
  p = grid.add(ponos::Point2(5.75, 5.75));
  p->velocity = ponos::vec2(0.01, 30);
  p = grid.add(ponos::Point2(5.25, 5.75));
  p->velocity = ponos::vec2(0.01, 40);
  grid.update();
  typename ZParticleGrid2D<FLIPParticle2D>::particle_iterator it2 =
      grid.getCell(ponos::ivec2(5, 5));
  std::cout << it2.count() << std::endl;
  {
    float vx = grid.gather(ParticleAttribute::VELOCITY_X, c, 0.51);
    float vy = grid.gather(ParticleAttribute::VELOCITY_Y, c, 0.51);
    std::cout << vx << " " << vy << std::endl;
    // CPPUNIT_ASSERT(IS_EQUAL(vx, 0.01) && IS_EQUAL(vy, 25.f));
  }
  for (typename ZParticleGrid2D<FLIPParticle2D>::particle_iterator it =
           grid.getCell(ponos::ivec2(5, 5));
       it.next(); ++it) {
    it.particleElement()->active = false;
  }
  {
    float vx = grid.gather(ParticleAttribute::VELOCITY_X, c, 0.51);
    float vy = grid.gather(ParticleAttribute::VELOCITY_Y, c, 0.51);
    std::cout << vx << " " << vy << std::endl;
    // CPPUNIT_ASSERT(IS_EQUAL(vx, 0.01) && IS_EQUAL(vy, 25.f));
  }
  grid.update();
  typename ZParticleGrid2D<FLIPParticle2D>::particle_iterator it =
      grid.getCell(ponos::ivec2(5, 5));
  std::cout << it.count() << std::endl;
  {
    float vx = grid.gather(ParticleAttribute::VELOCITY_X, c, 0.51);
    float vy = grid.gather(ParticleAttribute::VELOCITY_Y, c, 0.51);
    std::cout << vx << " " << vy << std::endl;
    // CPPUNIT_ASSERT(IS_EQUAL(vx, 0.01) && IS_EQUAL(vy, 25.f));
  }
  p = grid.add(ponos::Point2(5.25, 5.25));
  p->velocity = ponos::vec2(0.01, 10);
  p = grid.add(ponos::Point2(5.75, 5.25));
  p->velocity = ponos::vec2(0.01, 20);
  p = grid.add(ponos::Point2(5.75, 5.75));
  p->velocity = ponos::vec2(0.01, 30);
  p = grid.add(ponos::Point2(5.25, 5.75));
  p->velocity = ponos::vec2(0.01, 40);
  grid.update();
  {
    float vx = grid.gather(ParticleAttribute::VELOCITY_X, c, 0.51);
    float vy = grid.gather(ParticleAttribute::VELOCITY_Y, c, 0.51);
    std::cout << vx << " " << vy << std::endl;
    // CPPUNIT_ASSERT(IS_EQUAL(vx, 0.01) && IS_EQUAL(vy, 25.f));
  }
}
