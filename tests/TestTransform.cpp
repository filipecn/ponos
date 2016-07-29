#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>

using namespace ponos;

class TestTransform : public CppUnit::TestCase {
	public:
		CPPUNIT_TEST_SUITE(TestTransform);
		CPPUNIT_TEST(testOperator);
		CPPUNIT_TEST_SUITE_END();

		void testOperator();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestTransform);

void TestTransform::testOperator() {
	Transform t;
	Point3 p(1, 2, 3);
	Point3 tp = t(p);
	for(int i = 0; i < 3; i++)
		CPPUNIT_ASSERT(IS_EQUAL(p[i], tp[i]));
}
