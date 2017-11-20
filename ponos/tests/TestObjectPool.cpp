#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>

class TestObjectPool : public CppUnit::TestCase {
	public:
		CPPUNIT_TEST_SUITE(TestObjectPool);
		CPPUNIT_TEST(testCreate);
		CPPUNIT_TEST(testIterate);
		CPPUNIT_TEST(testDestroy);
		CPPUNIT_TEST(testStruct);
		CPPUNIT_TEST(testDynamic);
		CPPUNIT_TEST_SUITE_END();

		void testCreate();
		void testIterate();
		void testDestroy();
		void testStruct();
		void testDynamic();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestObjectPool);

void TestObjectPool::testCreate() {
	ponos::ObjectPool<int, 10> pool;
	CPPUNIT_ASSERT(pool.size() == 0);
	for(int i = 0; i < 10; i++) {
		int *p = pool.create();
		*p = i;
		CPPUNIT_ASSERT(p != nullptr);
		CPPUNIT_ASSERT(*p == i);
	}
	CPPUNIT_ASSERT(pool.size() == 10);
	int *p = pool.create();
	CPPUNIT_ASSERT(p == nullptr);
}

void TestObjectPool::testIterate() {
	ponos::ObjectPool<int, 10> pool;
	for(int i = 0; i < 10; i++) {
		int* p = pool.create();
		*p = i;
	}
	ponos::ObjectPool<int, 10>::iterator it(pool);
	int k = 0;
	while(it.next()) {
		int* p = *it;
		CPPUNIT_ASSERT(*p == k++);
		++it;
	}
}

void TestObjectPool::testDestroy() {
	ponos::ObjectPool<int, 10> pool;
	std::vector<int*> objs;
	for(int i = 0; i < 10; i++) {
		objs.push_back(pool.create());
		*objs[i] = i;
	}
	CPPUNIT_ASSERT(pool.size() == 10);
	for(int i = 0; i < 10; i++) {
		pool.destroy(objs[i]);
	}
	CPPUNIT_ASSERT(pool.size() == 0);
	return;
	objs.clear();
	for(int i = 0; i < 10; i++) {
		objs.push_back(pool.create());
		*objs[i] = i;
	}
	for(int i = 0; i < 10; i += 2)
		pool.destroy(objs[i]);
	ponos::ObjectPool<int, 10>::iterator it(pool);
	int k = 0;
	while(it.next()) {
		int* p = *it;
		CPPUNIT_ASSERT(*p == k);
		k += 2;
		++it;
	}
	CPPUNIT_ASSERT(pool.size() == 5);
}

struct S {
	int a; char b;
	double v[100];
	S(const int& _a, const char& _b) {
		a = _a; b = _b;
		for(size_t i = 0; i < 100; i++)
			v[i] = i;
	}
};

void TestObjectPool::testStruct() {
	ponos::ObjectPool<S, 100> pool;
	for(int i = 0; i < 100; i++) {
		pool.create(i, 'c');
	}
	ponos::ObjectPool<S, 100>::iterator it(pool);
	int k = 0;
	while(it.next()) {
		S* s = *it;
		CPPUNIT_ASSERT(s->a == k++);
		CPPUNIT_ASSERT(s->b == 'c');
		++it;
	}
}

void TestObjectPool::testDynamic() {
	ponos::DynamicObjectPool<S> pool;
	CPPUNIT_ASSERT(pool.size() == 0);
	std::vector<int> objs;
	for(int i = 0; i < 100; i++)
		objs.push_back(pool.create(i, 'c'));
	CPPUNIT_ASSERT(pool.size() == 100);
	{
		ponos::DynamicObjectPool<S>::iterator it(pool);
		int k = 0;
		while(it.next()) {
			S* s = *it;
			CPPUNIT_ASSERT(s->a == k++);
			CPPUNIT_ASSERT(s->b == 'c');
			++it;
		}
	}
	for(int i = 0; i < 50; i++)
		pool.destroy(objs[i]);
	CPPUNIT_ASSERT(pool.size() == 50);
	ponos::DynamicObjectPool<S>::iterator it(pool);
	int k = 50;
	while(it.next()) {
		S* s = *it;
		//std::cout << "checking " << s->a << " " << k << std::endl;
		CPPUNIT_ASSERT(s->a == k++);
		CPPUNIT_ASSERT(s->b == 'c');
		++it;
	}
}
