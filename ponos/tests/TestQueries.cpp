#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>

using namespace ponos;

class TestQueries : public CppUnit::TestCase {
	public:
		CPPUNIT_TEST_SUITE(TestQueries);
		CPPUNIT_TEST(testRaySegment);
		CPPUNIT_TEST_SUITE_END();

		void testRaySegment();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestQueries);

void TestQueries::testRaySegment() {
	Segment3 s(Point3(-15, 0, 0), Point3(10,0,0));
	Ray3 r(Point3(0, 20, 0), vec3(0, -1, 0));
	CPPUNIT_ASSERT(ray_segment_intersection(r, s));
	Ray3 r2(Point3(15,0,0), vec3(0, -1, 0));
	CPPUNIT_ASSERT(!ray_segment_intersection(r2, s));

	Ray3 r3(Point3(19.266, 1.79213, 5.06042), vec3(-0.896717, -0.296824, -0.32832));
	std::cout << r3(11.1965);
	Ray3 sr(s.a, s.b - s.a);
	std::cout << sr(0.969035);
	CPPUNIT_ASSERT(!ray_segment_intersection(r3, s));


	Ray3 r4(Point3(10.5076, 1.30321, 1.44688), vec3(-0.878212, -0.321182, -0.354382));
	CPPUNIT_ASSERT(ray_segment_intersection(r4, s));
}

