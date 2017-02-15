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
		CPPUNIT_TEST_SUITE_END();

		void testParticleIterator();
		void testUpdate();
		void testTree();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestZParticleGrid);

void TestZParticleGrid::testParticleIterator() {
	ZParticleGrid2D<FLIPParticle2D> grid(16, 16, ponos::scale(1, 1));
	std::vector<FLIPParticle2D> v;
	for(int i = 0; i < 16; i++)
		for(int j = 0; j < 16; j++)
			v.emplace_back(ponos::Point2(i, j));

	for(int i = 0; i < 16; i++)
		for(int j = 0; j < 16; j++)
			grid.add(ponos::Point2(i, j));

	size_t i = 0;
	for(ZParticleGrid2D<FLIPParticle2D>::particle_iterator it(grid); it.next(); ++it) {
		auto p = *it;
		CPPUNIT_ASSERT(v[i].position.x == p->position.x);
		CPPUNIT_ASSERT(v[i].position.y == p->position.y);
		i++;
	}
	CPPUNIT_ASSERT(i == 16 * 16);
}

void TestZParticleGrid::testUpdate() {
	ZParticleGrid2D<FLIPParticle2D> grid(8, 8, ponos::scale(1, 1));
	for(int i = 0; i < 8; i++)
		for(int j = 0; j < 8; j++)
			grid.add(ponos::Point2(i, j));
	grid.update();
	std::vector<ponos::Point2> order;
	int ddx = 0, ddy = 0;
	int dx = 0 + ddx, dy = 0 + ddy;
	order.emplace_back(0 + dx, 0 + dy);
	order.emplace_back(1 + dx, 0 + dy);
	order.emplace_back(0 + dx, 1 + dy);
	order.emplace_back(1 + dx, 1 + dy);
	dx = 2 + ddx; dy = 0 + ddy;
	order.emplace_back(0 + dx, 0 + dy);
	order.emplace_back(1 + dx, 0 + dy);
	order.emplace_back(0 + dx, 1 + dy);
	order.emplace_back(1 + dx, 1 + dy);
	dx = 0 + ddx; dy = 2 + ddy;
	order.emplace_back(0 + dx, 0 + dy);
	order.emplace_back(1 + dx, 0 + dy);
	order.emplace_back(0 + dx, 1 + dy);
	order.emplace_back(1 + dx, 1 + dy);
	dx = 2 + ddx; dy = 2 + ddy;
	order.emplace_back(0 + dx, 0 + dy);
	order.emplace_back(1 + dx, 0 + dy);
	order.emplace_back(0 + dx, 1 + dy);
	order.emplace_back(1 + dx, 1 + dy);

	ddx = 4; ddy = 0;
	dx = 0 + ddx; dy = 0 + ddy;
	order.emplace_back(0 + dx, 0 + dy);
	order.emplace_back(1 + dx, 0 + dy);
	order.emplace_back(0 + dx, 1 + dy);
	order.emplace_back(1 + dx, 1 + dy);
	dx = 2 + ddx; dy = 0 + ddy;
	order.emplace_back(0 + dx, 0 + dy);
	order.emplace_back(1 + dx, 0 + dy);
	order.emplace_back(0 + dx, 1 + dy);
	order.emplace_back(1 + dx, 1 + dy);
	dx = 0 + ddx; dy = 2 + ddy;
	order.emplace_back(0 + dx, 0 + dy);
	order.emplace_back(1 + dx, 0 + dy);
	order.emplace_back(0 + dx, 1 + dy);
	order.emplace_back(1 + dx, 1 + dy);
	dx = 2 + ddx; dy = 2 + ddy;
	order.emplace_back(0 + dx, 0 + dy);
	order.emplace_back(1 + dx, 0 + dy);
	order.emplace_back(0 + dx, 1 + dy);
	order.emplace_back(1 + dx, 1 + dy);

	ddx = 0; ddy = 4;
	dx = 0 + ddx; dy = 0 + ddy;
	order.emplace_back(0 + dx, 0 + dy);
	order.emplace_back(1 + dx, 0 + dy);
	order.emplace_back(0 + dx, 1 + dy);
	order.emplace_back(1 + dx, 1 + dy);
	dx = 2 + ddx; dy = 0 + ddy;
	order.emplace_back(0 + dx, 0 + dy);
	order.emplace_back(1 + dx, 0 + dy);
	order.emplace_back(0 + dx, 1 + dy);
	order.emplace_back(1 + dx, 1 + dy);
	dx = 0 + ddx; dy = 2 + ddy;
	order.emplace_back(0 + dx, 0 + dy);
	order.emplace_back(1 + dx, 0 + dy);
	order.emplace_back(0 + dx, 1 + dy);
	order.emplace_back(1 + dx, 1 + dy);
	dx = 2 + ddx; dy = 2 + ddy;
	order.emplace_back(0 + dx, 0 + dy);
	order.emplace_back(1 + dx, 0 + dy);
	order.emplace_back(0 + dx, 1 + dy);
	order.emplace_back(1 + dx, 1 + dy);

	ddx = 4; ddy = 4;
	dx = 0 + ddx; dy = 0 + ddy;
	order.emplace_back(0 + dx, 0 + dy);
	order.emplace_back(1 + dx, 0 + dy);
	order.emplace_back(0 + dx, 1 + dy);
	order.emplace_back(1 + dx, 1 + dy);
	dx = 2 + ddx; dy = 0 + ddy;
	order.emplace_back(0 + dx, 0 + dy);
	order.emplace_back(1 + dx, 0 + dy);
	order.emplace_back(0 + dx, 1 + dy);
	order.emplace_back(1 + dx, 1 + dy);
	dx = 0 + ddx; dy = 2 + ddy;
	order.emplace_back(0 + dx, 0 + dy);
	order.emplace_back(1 + dx, 0 + dy);
	order.emplace_back(0 + dx, 1 + dy);
	order.emplace_back(1 + dx, 1 + dy);
	dx = 2 + ddx; dy = 2 + ddy;
	order.emplace_back(0 + dx, 0 + dy);
	order.emplace_back(1 + dx, 0 + dy);
	order.emplace_back(0 + dx, 1 + dy);
	order.emplace_back(1 + dx, 1 + dy);

	CPPUNIT_ASSERT(order.size() == 8 * 8);
	int i = 0;
	for(ZParticleGrid2D<FLIPParticle2D>::particle_iterator it(grid); it.next(); ++it) {
		auto p = *it;
		CPPUNIT_ASSERT(order[i].x == p->position.x);
		CPPUNIT_ASSERT(order[i].y == p->position.y);
		i++;
	}
}

void TestZParticleGrid::testTree() {
	ZParticleGrid2D<FLIPParticle2D> grid(8, 8, ponos::scale(1, 1));
	for(int i = 0; i < 8; i++)
		for(int j = 0; j < 8; j++)
			grid.add(ponos::Point2(i, j));

	ZParticleGrid2D<FLIPParticle2D>::tree tree(grid, [&grid](uint32 id, uint32 depth) {
			ZParticleGrid2D<FLIPParticle2D>::particle_iterator it(grid, id, depth);
			std::cout << it.count() << std::endl;
			return true;
			});
}
