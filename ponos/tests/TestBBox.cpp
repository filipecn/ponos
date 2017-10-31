#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>


class TestBBox : public CppUnit::TestCase {
	public:
		CPPUNIT_TEST_SUITE(TestBBox);
		CPPUNIT_TEST(testUnion);
		CPPUNIT_TEST_SUITE_END();

		void testUnion();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestBBox);

void TestBBox::testUnion() {
	CPPUNIT_ASSERT(true);
}

