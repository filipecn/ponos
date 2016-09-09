#include <ponos.h>
#include <poseidon.h>
#include <cppunit/extensions/HelperMacros.h>

using namespace poseidon;

class TestVDBParticleGrid : public CppUnit::TestCase {
	public:
		CPPUNIT_TEST_SUITE(TestVDBParticleGrid);
		CPPUNIT_TEST(testAddParticle);
		CPPUNIT_TEST(testIterateCell);
		CPPUNIT_TEST(testWrite);
		CPPUNIT_TEST(testParticleCount);
		CPPUNIT_TEST(testIterateNeighbours);
		CPPUNIT_TEST(testTransforms);
		CPPUNIT_TEST(testComputeDensity);
		CPPUNIT_TEST_SUITE_END();

		void testAddParticle();
		void testIterateCell();
		void testWrite();
		void testParticleCount();
		void testIterateNeighbours();
		void testTransforms();
		void testComputeDensity();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestVDBParticleGrid);

void TestVDBParticleGrid::testIterateNeighbours() {
	VDBParticleGrid grid(ponos::ivec3(4, 4, 4), 1.f, ponos::vec3());

	for(int x = 2; x < 5; x++)
		for(int y = 2; y < 5; y++)
			for(int z = 2; z < 5; z++) {
				grid.addParticle(ponos::ivec3(x, y, z), ponos::Point3(0.4, 0, 0), ponos::vec3());
				grid.addParticle(ponos::ivec3(x, y, z), ponos::Point3(0, 0.4, 0), ponos::vec3());
				grid.addParticle(ponos::ivec3(x, y, z), ponos::Point3(0, 0, 0.4), ponos::vec3());
				grid.addParticle(ponos::ivec3(x, y, z), ponos::Point3(0, 0,   0), ponos::vec3());
			}

	for(int x = 2; x < 3; x++)
		for(int y = 2; y < 3; y++)
			for(int z = 2; z < 3; z++) {
				int count = 0;
				grid.iterateCellNeighbours(ponos::ivec3(x, y, z), ponos::ivec3(0, 0, 0),
						[&count](const size_t& id){count++;});
				CPPUNIT_ASSERT(count == 4);
				count = 0;
				grid.iterateCell(ponos::ivec3(x, y, z), [&count](const size_t& id){count++;});
				CPPUNIT_ASSERT(count == 4);
				CPPUNIT_ASSERT(grid.particleCount(ponos::ivec3(x, y, z)) == 4);
			}

	int count = 0;
	grid.iterateCellNeighbours(ponos::ivec3(3, 3, 3), ponos::ivec3(1, 1, 1),
			[&count](const size_t& id){count++;});
	CPPUNIT_ASSERT(count == 2 * 2 * 2 * 4);
	count = 0;
	grid.iterateCellNeighbours(ponos::ivec3(2, 2, 1), ponos::ivec3(1, 1, 1),
			[&count](const size_t& id){count++;});
	CPPUNIT_ASSERT(count == 2 * 2 * 1 * 4);
}

void TestVDBParticleGrid::testAddParticle() {
	VDBParticleGrid grid(ponos::ivec3(10, 10, 10), 3.f, ponos::vec3());
	// point 0
	grid.addParticle(ponos::ivec3(2, 2, 2), ponos::Point3(-0.4, 0, 0), ponos::vec3());
	// point 1
	grid.addParticle(ponos::ivec3(2, 2, 2), ponos::Point3(0.4, 0, 0.4), ponos::vec3());
	// point 2
	grid.addParticle(ponos::ivec3(3, 2, 2), ponos::Point3(-0.4, 0, 0), ponos::vec3());
	// point 3
	grid.addParticle(ponos::ivec3(3, 2, 2), ponos::Point3(-0.4, 0, 0.2), ponos::vec3());
	// point 4
	grid.addParticle(ponos::ivec3(3, 2, 2), ponos::Point3(-0.4, 0.1, 0.2), ponos::vec3());
	// point 5
	grid.addParticle(ponos::ivec3(2, 1, 1), ponos::Point3(-0.4, 0, 0), ponos::vec3());

	grid.init();

	CPPUNIT_ASSERT(grid.particleCount() == 6);

	int count = 0;
	grid.iterateCell(ponos::ivec3(3, 2, 2), [&count](int id) { count++; });
	CPPUNIT_ASSERT(count == 3);
	count = 0;
	grid.iterateCell(ponos::ivec3(2, 2, 2), [&count](int id) { count++; });
	CPPUNIT_ASSERT(count == 2);
	count = 0;
	grid.iterateCell(ponos::ivec3(2, 1, 1), [&count](int id) { count++; });
	CPPUNIT_ASSERT(count == 1);
}

void TestVDBParticleGrid::testIterateCell() {
	VDBParticleGrid grid(ponos::ivec3(10, 10, 10), 1.f, ponos::vec3());
	// point 0
	grid.addParticle(ponos::Point3(0, 0, 0), ponos::vec3(0, 1, 0));
	// point 1
	grid.addParticle(ponos::Point3(1, 1, 1), ponos::vec3(0, 1, 0));
	// point 2
	grid.addParticle(ponos::Point3(2, 2, 2), ponos::vec3(0, 1, 0));
	// point 3
	grid.addParticle(ponos::Point3(2, 2.3, 2), ponos::vec3(0, 1, 0));
	// point 4
	grid.addParticle(ponos::Point3(2.4, 2, 2.4), ponos::vec3(0, 1, 0));
	// point 5
	grid.addParticle(ponos::Point3(3, 3, 3), ponos::vec3(0, 1, 0));
	// point 6
	grid.addParticle(ponos::Point3(4, 4, 4), ponos::vec3(0, 1, 0));

	grid.init();

	CPPUNIT_ASSERT(grid.particleCount() == 7);

	int expected[] = { 2, 3, 4 };
	std::vector<int> found;
	grid.iterateCell(ponos::ivec3(2, 2, 2), [&found](int id) { found.push_back(id); });
	CPPUNIT_ASSERT(found.size() == 3);
	for(int i = 0 ; i < 3; i++)
		CPPUNIT_ASSERT(found[i] == expected[i]);
}

void TestVDBParticleGrid::testWrite() {
	VDBParticleGrid grid(ponos::ivec3(10, 10, 10), 1.f, ponos::vec3());
	// point 0
	grid.addParticle(ponos::Point3(0, 0, 0), ponos::vec3(0, 1, 0));
	// point 1
	grid.addParticle(ponos::Point3(1, 1, 1), ponos::vec3(0, 1, 0));
	// point 2
	grid.addParticle(ponos::Point3(2, 2, 2), ponos::vec3(0, 1, 0));
	// point 3
	grid.addParticle(ponos::Point3(2, 2.3, 2), ponos::vec3(0, 1, 0));
	// point 4
	grid.addParticle(ponos::Point3(2.4, 2, 2.4), ponos::vec3(0, 1, 0));
	// point 5
	grid.addParticle(ponos::Point3(3, 3, 3), ponos::vec3(0, 1, 0));
	// point 6
	grid.addParticle(ponos::Point3(4, 4, 4), ponos::vec3(0, 1, 0));

	grid.init();

	CPPUNIT_ASSERT(grid.particleCount() == 7);

	CPPUNIT_ASSERT(grid.particleCount(ponos::ivec3(1, 1, 1)) == 1);

	int expected[] = { 2, 3, 4 };
	std::vector<int> found;
	grid.iterateCell(ponos::ivec3(2, 2, 2), [&found](int id) { found.push_back(id); });
	CPPUNIT_ASSERT(found.size() == 3);
	for(int i = 0 ; i < 3; i++)
		CPPUNIT_ASSERT(found[i] == expected[i]);

	grid.setParticle(2, ponos::Point3(1.2, 1, 1), ponos::vec3());
	grid.setParticle(3, ponos::Point3(1.2, 1.4, 1), ponos::vec3());
	grid.setParticle(4, ponos::Point3(0.5, 1, 1.4), ponos::vec3());

	CPPUNIT_ASSERT(grid.particleCount(ponos::ivec3(2, 2, 2)) == 0);
	CPPUNIT_ASSERT(grid.particleCount(ponos::ivec3(1, 1, 1)) == 4);
}

void TestVDBParticleGrid::testParticleCount() {
	VDBParticleGrid grid(ponos::ivec3(10, 10, 10), 1.f, ponos::vec3());
	// point 0
	grid.addParticle(ponos::Point3(0, 0, 0), ponos::vec3(0, 1, 0));
	// point 1
	grid.addParticle(ponos::Point3(1, 1, 1), ponos::vec3(0, 1, 0));
	// point 2
	grid.addParticle(ponos::Point3(2, 2, 2), ponos::vec3(0, 1, 0));
	// point 3
	grid.addParticle(ponos::Point3(2, 2.3, 2), ponos::vec3(0, 1, 0));
	// point 4
	grid.addParticle(ponos::Point3(2.4, 2, 2.4), ponos::vec3(0, 1, 0));
	// point 5
	grid.addParticle(ponos::Point3(3, 3, 3), ponos::vec3(0, 1, 0));
	// point 6
	grid.addParticle(ponos::Point3(4, 4, 4), ponos::vec3(0, 1, 0));
	// point 7
	grid.addParticle(ponos::Point3(-4, 4, 4), ponos::vec3(0, 1, 0));
	// point 8
	grid.addParticle(ponos::Point3(4, 14, 4), ponos::vec3(0, 1, 0));

	grid.init();

	CPPUNIT_ASSERT(grid.particleCount() == 7);

	CPPUNIT_ASSERT(grid.particleCount(ponos::ivec3(1, 1, 1)) == 1);
	CPPUNIT_ASSERT(grid.particleCount(ponos::ivec3(2, 2, 2)) == 3);
	CPPUNIT_ASSERT(grid.particleCount(ponos::ivec3(4, 4, 4)) == 1);
}

void TestVDBParticleGrid::testTransforms() {
	VDBParticleGrid grid(ponos::ivec3(10, 10, 10), 2.f, ponos::vec3());
	ponos::Point3 wp(4.2f, 8.4f, 0.0f);
	ponos::Point3 gp = grid.toGrid(wp);
	CPPUNIT_ASSERT(gp == ponos::Point3(2.1f, 4.2f, 0.f));
	ponos::Point3 vp = grid.worldToVoxel(wp);
	CPPUNIT_ASSERT(vp == ponos::Point3(0.1f, 0.2f, 0.f));
	ponos::ivec3 ip = grid.worldToIndex(wp);
	CPPUNIT_ASSERT(ip == ponos::ivec3(2, 4, 0));

	gp = ponos::Point3(8.1, 10.2, 20.3);
	CPPUNIT_ASSERT(grid.toWorld(gp) == ponos::Point3(16.2, 20.4, 40.6));
	CPPUNIT_ASSERT(grid.gridToIndex(gp) == ponos::ivec3(8, 10, 20));
	CPPUNIT_ASSERT(grid.gridToVoxel(gp) == ponos::Point3(0.1, 0.2, 0.3));
}

void TestVDBParticleGrid::testComputeDensity() {
	VDBParticleGrid grid(ponos::ivec3(10, 10, 10), 1.f, ponos::vec3());
	grid.particleMass = 1.f;
	float density = 1.f;
	float max_density = 1.f;
	ponos::ivec3 ijk;
	float h = density / 10.f;
	int &x = ijk[0], &y = ijk[1], &z = ijk[2];
	for(x = 0; x < 10; x++)
		for(y = 0; y < 10; y++)
			for(z = 0; z < 10; z++)
				grid.addParticle(ponos::Point3(x, y, z) * h, ponos::vec3());
	grid.computeDensity(density, max_density);
}
