#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>

#include <vector>

class TestSearch : public CppUnit::TestCase {
	public:
		CPPUNIT_TEST_SUITE(TestSearch);
		CPPUNIT_TEST(testFound);
		CPPUNIT_TEST(testFoundMultiple);
		CPPUNIT_TEST(testNotFound);
		CPPUNIT_TEST(testFirstGreater);
		CPPUNIT_TEST(testLastSmaller);
		CPPUNIT_TEST_SUITE_END();

		void testFound();
		void testFoundMultiple();
		void testNotFound();
		void testFirstGreater();
		void testLastSmaller();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestSearch);

void TestSearch::testFound() {
	std::vector<int> v;
	for(int i = 0; i < 10; i++)
		v.emplace_back(i);
	for(int i = 0; i < 10; i++)
		CPPUNIT_ASSERT(ponos::binary_search<int>(&v[0], v.size(), i) == i);
}

void TestSearch::testFoundMultiple() {
	int v[10] = {1, 2, 3, 3, 3, 3, 4, 4, 5, 5};
	CPPUNIT_ASSERT(ponos::binary_search<int>(v, 10, 1) == 0);
	CPPUNIT_ASSERT(ponos::binary_search<int>(v, 10, 2) == 1);
	CPPUNIT_ASSERT(ponos::binary_search<int>(v, 10, 3) == 2);
	CPPUNIT_ASSERT(ponos::binary_search<int>(v, 10, 4) == 6);
	CPPUNIT_ASSERT(ponos::binary_search<int>(v, 10, 5) == 8);
}

void TestSearch::testNotFound() {
	int v[10] = {1, 2, 3, 3, 3, 3, 5, 5, 7, 7};
	CPPUNIT_ASSERT(ponos::binary_search<int>(v, 10, 6) == -1);
	CPPUNIT_ASSERT(ponos::binary_search<int>(v, 10, 8) == -1);
	CPPUNIT_ASSERT(ponos::binary_search<int>(v, 10, 0) == -1);
}

void TestSearch::testFirstGreater() {
	int v[10] = {1, 2, 3, 3, 3, 3, 4, 4, 6, 6};
	CPPUNIT_ASSERT(ponos::upper_bound<int>(v, 10, 0) == 0);
	CPPUNIT_ASSERT(ponos::upper_bound<int>(v, 10, 1) == 1);
	CPPUNIT_ASSERT(ponos::upper_bound<int>(v, 10, 2) == 2);
	CPPUNIT_ASSERT(ponos::upper_bound<int>(v, 10, 4) == 8);
	CPPUNIT_ASSERT(ponos::upper_bound<int>(v, 10, 5) == 8);
	CPPUNIT_ASSERT(ponos::upper_bound<int>(v, 10, 7) == 10);
}

void TestSearch::testLastSmaller() {
	int v[10] = {1, 2, 3, 3, 3, 3, 4, 4, 6, 6};
	CPPUNIT_ASSERT(ponos::lower_bound<int>(v, 10, 6) == 7);
	CPPUNIT_ASSERT(ponos::lower_bound<int>(v, 10, 6) == 7);
	CPPUNIT_ASSERT(ponos::lower_bound<int>(v, 10, 5) == 7);
	CPPUNIT_ASSERT(ponos::lower_bound<int>(v, 10, 4) == 5);
	CPPUNIT_ASSERT(ponos::lower_bound<int>(v, 10, 3) == 1);
	CPPUNIT_ASSERT(ponos::lower_bound<int>(v, 10, 2) == 0);
	CPPUNIT_ASSERT(ponos::lower_bound<int>(v, 10, 1) == -1);
}
