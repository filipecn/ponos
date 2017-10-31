#include <hercules.h>
#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>

using namespace ponos;
using namespace hercules::cds;

class TestAABBGrid : public CppUnit::TestCase {
	public:
		CPPUNIT_TEST_SUITE(TestAABBGrid);
		CPPUNIT_TEST(testCreate);
		CPPUNIT_TEST(testDestroy);
		CPPUNIT_TEST(testIndexPtr);
		CPPUNIT_TEST_SUITE_END();

		void testCreate();
		void testDestroy();
		void testIndexPtr();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestAABBGrid);

class ActiveClass : AABBObjectInterface<ponos::BBox2D> {
	public:
    ActiveClass() { std::cout << "damn\n"; }
		ActiveClass(BBox2D b)
			: bb(b) {}
		bool isActive() { return true; }
		BBox2D getWBBox() { return bb; }
    void destroy() {}
	private:
		BBox2D bb;
};

void TestAABBGrid::testCreate() {
	AABBGrid2D<ActiveClass> grid;
	for(int i = 0; i < 10; i++)
		grid.create(BBox2D(ponos::Point2(i, i), ponos::Point2(i + 2, i + 2)));
	CPPUNIT_ASSERT(grid.size() == 10);
	int k = 0;
	for(AABBGrid2D<ActiveClass>::iterator it(grid); it.next(); ++it) {
		BBox2D b = (*it)->getWBBox();
		CPPUNIT_ASSERT(b.pMin == Point2(k, k));
		CPPUNIT_ASSERT(b.pMax == Point2(k + 2, k + 2));
		k++;
	}
	CPPUNIT_ASSERT(k == 10);
}

void TestAABBGrid::testDestroy() {
	AABBGrid2D<ActiveClass> grid;
	std::vector<IndexPointer<ActiveClass> > ptrs;
	for(int i = 0; i < 10; i++)
		ptrs.push_back(grid.create(BBox2D(ponos::Point2(i, i), ponos::Point2(i + 2, i + 2))));
	CPPUNIT_ASSERT(grid.size() == 10);
	int k = 0;
	for(AABBGrid2D<ActiveClass>::iterator it(grid); it.next(); ++it) {
		BBox2D b = (*it)->getWBBox();
		CPPUNIT_ASSERT(b.pMin == Point2(k, k));
		CPPUNIT_ASSERT(b.pMax == Point2(k + 2, k + 2));
		k++;
	}
	CPPUNIT_ASSERT(k == 10);
	for(int i = 0; i < 10; i++)
		if(i % 2)
			grid.destroy(ptrs[i]);
	CPPUNIT_ASSERT(grid.size() == 5);
	k = 0;
	for(AABBGrid2D<ActiveClass>::iterator it(grid); it.next(); ++it) {
		BBox2D b = (*it)->getWBBox();
		CPPUNIT_ASSERT(static_cast<int>(b.pMin[0]) % 2 == 0);
		CPPUNIT_ASSERT(static_cast<int>(b.pMax[0]) % 2 == 0);
		CPPUNIT_ASSERT(static_cast<int>(b.pMin[1]) % 2 == 0);
		CPPUNIT_ASSERT(static_cast<int>(b.pMax[1]) % 2 == 0);
		k++;
	}
	CPPUNIT_ASSERT(k == 5);
}

void TestAABBGrid::testIndexPtr() {
	AABBGrid2D<ActiveClass> grid;
	std::vector<IndexPointer<ActiveClass> > ptrs;
	for(int i = 0; i < 10; i++)
		ptrs.push_back(grid.create(BBox2D(ponos::Point2(i, i), ponos::Point2(i + 2, i + 2))));
	CPPUNIT_ASSERT(grid.size() == 10);
	int k = 0;
	for(auto ptr : ptrs) {
		BBox2D bb = ptr->getWBBox();
		CPPUNIT_ASSERT(static_cast<int>(bb.pMin[0]) == k);
		CPPUNIT_ASSERT(static_cast<int>(bb.pMin[1]) == k);
		CPPUNIT_ASSERT(static_cast<int>(bb.pMax[0]) == k + 2);
		CPPUNIT_ASSERT(static_cast<int>(bb.pMax[1]) == k + 2);
		k++;
	}
}
