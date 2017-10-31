#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>

using namespace ponos;

class TestTransform : public CppUnit::TestCase {
	public:
		CPPUNIT_TEST_SUITE(TestTransform);
		CPPUNIT_TEST(testOperator);
		CPPUNIT_TEST(testRotate);
		CPPUNIT_TEST_SUITE_END();

		void testOperator();
		void testRotate();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestTransform);

void TestTransform::testOperator() {
	Transform t;
	Point3 p(1, 2, 3);
	Point3 tp = t(p);
	for(int i = 0; i < 3; i++)
		CPPUNIT_ASSERT(IS_EQUAL(p[i], tp[i]));
}

void TestTransform::testRotate() {
	Transform tx = rotateX(90.f);
	Transform ty = rotateY(90.f);
	Transform tz = rotateZ(90.f);
	ponos::vec3 v(0, 1, 0);
	ponos::vec3 rx(0, 0, 1);
	ponos::vec3 ry(0, 1, 0);
	ponos::vec3 rz(-1, 0, 0);
	CPPUNIT_ASSERT(rx == tx(v));
	CPPUNIT_ASSERT(ry == ty(v));
	CPPUNIT_ASSERT(rz == tz(v));
	Transform gx = rotate(90.f, vec3(1, 0, 0));
	CPPUNIT_ASSERT(rx == gx(v));
	Transform gy = rotate(90.f, vec3(0, 1, 0));
	CPPUNIT_ASSERT(ry == gy(v));
	Transform gz = rotate(90.f, vec3(0, 0, 1));
	CPPUNIT_ASSERT(rz == gz(v));
	Transform t = rotateX(45.f);
	Transform g = rotate(45.f, vec3(1,0,0));
	CPPUNIT_ASSERT(t == g);
	t = rotateX(60.f);
	g = rotate(60.f, vec3(1,0,0));
	CPPUNIT_ASSERT(t == g);
	t = rotateY(60.f);
	g = rotate(60.f, vec3(0,1,0));
	CPPUNIT_ASSERT(t == g);
	t = rotateZ(60.f);
	g = rotate(60.f, vec3(0,0,1));
	CPPUNIT_ASSERT(t == g);
}
