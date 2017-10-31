#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>

#include <vector>

class TestSort : public CppUnit::TestCase {
	public:
		CPPUNIT_TEST_SUITE(TestSort);
		CPPUNIT_TEST(testInsertion);
		CPPUNIT_TEST_SUITE_END();

		void testInsertion();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestSort);

void TestSort::testInsertion() {
	std::vector<int> v;
	for(int i = 23534; i >= 0; i--)
		v.push_back(i);
	for(size_t i = 1; i < v.size(); i++)
		CPPUNIT_ASSERT(v[i - 1] > v[i]);
	ponos::insertion_sort(&v[0], v.size());
	for(size_t i = 1; i < v.size(); i++)
		CPPUNIT_ASSERT(v[i - 1] < v[i]);
	ponos::insertion_sort<int>(&v[0], v.size(),
			[](const int& a, const int& b) {return b < a;});
	for(size_t i = 1; i < v.size(); i++)
		CPPUNIT_ASSERT(v[i - 1] > v[i]);
}

