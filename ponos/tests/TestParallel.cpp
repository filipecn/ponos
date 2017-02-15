#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>

using namespace ponos;

class TestParallel : public CppUnit::TestCase {
	public:
		CPPUNIT_TEST_SUITE(TestParallel);
		CPPUNIT_TEST(testFor);
		CPPUNIT_TEST(testForGrain);
		CPPUNIT_TEST_SUITE_END();

		void testFor();
		void testForGrain();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestParallel);

void TestParallel::testFor() {
	std::vector<int> v(100000, 1);
	for(size_t i = 0; i < v.size(); i++)
		CPPUNIT_ASSERT(v[i] == 1);
	ponos::parallel_for(0, 100000, [&v](size_t f, size_t l) { for(size_t i = f; i <= l; i++) v[i] = -1; });
	for(size_t i = 0; i < v.size(); i++)
		CPPUNIT_ASSERT(v[i] == -1);
}

void TestParallel::testForGrain() {
	std::vector<int> v(100000, 1);
	for(size_t i = 0; i < v.size(); i++)
		CPPUNIT_ASSERT(v[i] == 1);
	ponos::parallel_for(0, 100000, [&v](size_t f, size_t l) { for(size_t i = f; i <= l; i++) v[i] = -1; }, 100);
	for(size_t i = 0; i < v.size(); i++)
		CPPUNIT_ASSERT(v[i] == -1);
}
