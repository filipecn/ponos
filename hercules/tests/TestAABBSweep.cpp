#include <hercules.h>
#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>

using namespace ponos;
using namespace hercules::cds;

class TestAABBSweep : public CppUnit::TestCase {
	public:
		CPPUNIT_TEST_SUITE(TestAABBSweep);
		CPPUNIT_TEST(testCreate);
		CPPUNIT_TEST(testDestroy);
		CPPUNIT_TEST(testIndexPtr);
		CPPUNIT_TEST(testSortAndSweep);
		CPPUNIT_TEST_SUITE_END();

		void testCreate();
		void testDestroy();
		void testIndexPtr();
		void testSortAndSweep();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestAABBSweep);

class ActiveClass : AABBObjectInterface<ponos::BBox2D> {
	public:
		ActiveClass() {}
		ActiveClass(BBox2D b)
			: bb(b) {}
		bool isActive() { return true; }
		BBox2D getWBBox() { return bb; }
	private:
		BBox2D bb;
};

void TestAABBSweep::testCreate() {
	AABBSweep<ActiveClass> Sweep;
	for(int i = 0; i < 10; i++)
		Sweep.create(BBox2D(ponos::Point2(i, i), ponos::Point2(i + 2, i + 2)));
	CPPUNIT_ASSERT(Sweep.size() == 10);
	int k = 0;
	for(AABBSweep<ActiveClass>::iterator it(Sweep); it.next(); ++it) {
		BBox2D b = (*it)->getWBBox();
		CPPUNIT_ASSERT(b.pMin == Point2(k, k));
		CPPUNIT_ASSERT(b.pMax == Point2(k + 2, k + 2));
		k++;
	}
	CPPUNIT_ASSERT(k == 10);
}

void TestAABBSweep::testDestroy() {
	AABBSweep<ActiveClass> Sweep;
	std::vector<IndexPointer<ActiveClass> > ptrs;
	for(int i = 0; i < 10; i++)
		ptrs.push_back(Sweep.create(BBox2D(ponos::Point2(i, i), ponos::Point2(i + 2, i + 2))));
	CPPUNIT_ASSERT(Sweep.size() == 10);
	int k = 0;
	for(AABBSweep<ActiveClass>::iterator it(Sweep); it.next(); ++it) {
		BBox2D b = (*it)->getWBBox();
		CPPUNIT_ASSERT(b.pMin == Point2(k, k));
		CPPUNIT_ASSERT(b.pMax == Point2(k + 2, k + 2));
		k++;
	}
	CPPUNIT_ASSERT(k == 10);
	for(int i = 0; i < 10; i++)
		if(i % 2)
			Sweep.destroy(ptrs[i]);
	CPPUNIT_ASSERT(Sweep.size() == 5);
	k = 0;
	for(AABBSweep<ActiveClass>::iterator it(Sweep); it.next(); ++it) {
		BBox2D b = (*it)->getWBBox();
		CPPUNIT_ASSERT(static_cast<int>(b.pMin[0]) % 2 == 0);
		CPPUNIT_ASSERT(static_cast<int>(b.pMax[0]) % 2 == 0);
		CPPUNIT_ASSERT(static_cast<int>(b.pMin[1]) % 2 == 0);
		CPPUNIT_ASSERT(static_cast<int>(b.pMax[1]) % 2 == 0);
		k++;
	}
	CPPUNIT_ASSERT(k == 5);
}

void TestAABBSweep::testIndexPtr() {
	AABBSweep<ActiveClass> Sweep;
	std::vector<IndexPointer<ActiveClass> > ptrs;
	for(int i = 0; i < 10; i++)
		ptrs.push_back(Sweep.create(BBox2D(ponos::Point2(i, i), ponos::Point2(i + 2, i + 2))));
	CPPUNIT_ASSERT(Sweep.size() == 10);
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

void TestAABBSweep::testSortAndSweep() {
	AABBSweep<ActiveClass> Sweep;
	std::vector<IndexPointer<ActiveClass> > ptrs;
	for(int i = 0; i < 10; i++)
		Sweep.create(BBox2D(ponos::Point2(i, i), ponos::Point2(i + 2, i + 2)));
	CPPUNIT_ASSERT(Sweep.size() == 10);
	std::vector<Contact2D> contacts = Sweep.collide();
	CPPUNIT_ASSERT(contacts.size() == 10);
}

