#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>

using namespace ponos;

class TestMatrix : public CppUnit::TestCase {
	public:
		CPPUNIT_TEST_SUITE(TestMatrix);
		CPPUNIT_TEST(testInverse);
		CPPUNIT_TEST_SUITE_END();

		void testInverse();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestMatrix);

void print(const Matrix4x4& m) {
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++)
			std::cout << m.m[i][j] << " ";
		std::cout << std::endl;
	}
}

bool isIdentity(const Matrix4x4& m) {
	for(int i = 0; i < 4; i++)
		for(int j = 0; j < 4; j++)
			if(i != j && !IS_EQUAL(m.m[i][j], 0.f))
				return false;
			else if(i == j && !IS_EQUAL(m.m[i][j], 1.f))
				return false;
	return true;
}

void TestMatrix::testInverse() {
	Matrix4x4 m;
	CPPUNIT_ASSERT(isIdentity(m));
	Matrix4x4 mm = m * inverse(m);
	CPPUNIT_ASSERT(isIdentity(mm));
	m.m[0][0] = 0;
	m.m[1][0] = 0;
	m.m[2][0] = -1;
	m.m[3][0] = 0;

	m.m[0][1] = 0;
	m.m[1][1] = 1;
	m.m[2][1] = 0;
	m.m[3][1] = 0;

	m.m[0][2] = 1;
	m.m[1][2] = 0;
	m.m[2][2] = 0;
	m.m[3][2] = -1;

	m.m[0][3] = 0;
	m.m[1][3] = 0;
	m.m[2][3] = 0;
	m.m[3][3] = 1;
	print(m);
	print(inverse(m));
	mm = m * inverse(m);
	CPPUNIT_ASSERT(isIdentity(mm));
}
