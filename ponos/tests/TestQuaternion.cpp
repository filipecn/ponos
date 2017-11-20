#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>

using namespace ponos;

class TestQuaternion : public CppUnit::TestCase {
	public:
		CPPUNIT_TEST_SUITE(TestQuaternion);
		CPPUNIT_TEST(testConstructors);
		CPPUNIT_TEST_SUITE_END();

		void testConstructors();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestQuaternion);

void TestQuaternion::testConstructors() {
	Transform t = rotateY(60.f);
	quat qx;
	qx.fromAxisAndAngle(vec3(0,1,0), 60.f);
	quat q = quat(t);
	Transform tt = q.toTransform();
	quat qq(tt);
	//std::cout << q << qq;
	//std::cout << t.matrix() << tt.matrix();

	CPPUNIT_ASSERT(t == tt);
	CPPUNIT_ASSERT(q == qq);
}
