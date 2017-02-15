#include <ponos.h>
#include <poseidon.h>
#include <cppunit/extensions/HelperMacros.h>

using namespace poseidon;

class TestVDBMacGrid : public CppUnit::TestCase {
	public:
		CPPUNIT_TEST_SUITE(TestVDBMacGrid);
		CPPUNIT_TEST(testWrite);
		CPPUNIT_TEST_SUITE_END();

		void testWrite();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestVDBMacGrid);

void TestVDBMacGrid::testWrite() {
	VDBGrid grid(ponos::ivec3(), 1.f, 0.5f, ponos::vec3());

	ponos::ivec3 ijk;
	int& i = ijk[0], &j = ijk[1], &k = ijk[2];;
	for(i = 0; i < 10; i++)
		for(j = 0; j < 10; j++)
			for(k = 0; k < 10; k++)
				grid.set(ijk, 2.f);

	for(i = 0; i < 10; i++)
		for(j = 0; j < 10; j++)
			for(k = 0; k < 10; k++)
				CPPUNIT_ASSERT(IS_EQUAL(grid(ijk), 2.f));
}
