#include <ponos.h>
#include <poseidon.h>
#include <cppunit/extensions/HelperMacros.h>

using namespace poseidon;

class TestVDBParticleGrid : public CppUnit::TestCase {
	public:
		CPPUNIT_TEST_SUITE(TestVDBParticleGrid);
		CPPUNIT_TEST(testWrite);
		CPPUNIT_TEST_SUITE_END();

		void testWrite();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestVDBParticleGrid);

void TestVDBParticleGrid::testWrite() {
	VDBParticleGrid grid(ponos::ivec3(), 0.5f, ponos::vec3());

	std::vector<ponos::Point3> positions;
	positions.emplace_back(0, 0, 0);
	positions.emplace_back(1, 0, 0);
	positions.emplace_back(0, 1, 0);
	positions.emplace_back(0, 0, 1);
	positions.emplace_back(1, 1, 0);
	positions.emplace_back(1, 0, 1);

	std::vector<ponos::vec3> velocities;
	velocities.emplace_back(0, 1, 0);
	velocities.emplace_back(0, 1, 0);
	velocities.emplace_back(0, 1, 0);
	velocities.emplace_back(0, 1, 0);
	velocities.emplace_back(0, 1, 0);
	velocities.emplace_back(0, 1, 0);

	for(size_t i = 0; i < positions.size(); i++)
		grid.addParticle(positions[i], velocities[i]);

	grid.init();
	//CPPUNIT_ASSERT(IS_EQUAL(grid(ijk), 2.f));
}
