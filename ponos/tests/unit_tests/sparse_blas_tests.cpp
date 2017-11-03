#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;

TEST(SparseBlas, setV) {
  svecd a(200, 100);
  for (size_t i = 0; i < 200; i += 2)
    a.insert(i, 0.0);
  EXPECT_EQ(a.elementCount(), 100u);
  int count = 0;
  a.iterate([&](double &v, size_t i) {
    EXPECT_EQ(i % 2u, 0u);
    count++;
    ASSERT_NEAR(v, 0.0, 1e-8);
  });
  EXPECT_EQ(count, 100);
  SparseBlas2d::set(10.0, &a);
  EXPECT_EQ(a.elementCount(), 100u);
  a.iterate([&](double &v, size_t i) {
    EXPECT_EQ(i % 2u, 0u);
    count++;
    ASSERT_NEAR(v, 10.0, 1e-8);
  });
}

TEST(SparseBlas, setVV) {
  svecd a(100, 100), b(1, 1);
  for (size_t i = 0; i < 100; i++)
    a.insert(i, 10.0);
  SparseBlas2d::set(a, &b);
  EXPECT_EQ(a.elementCount(), b.elementCount());
  EXPECT_EQ(a.size(), b.size());
  svecd::const_iterator ait(a);
  svecd::const_iterator bit(b);
  while (ait.next() && bit.next()) {
    ASSERT_NEAR(ait.value(), bit.value(), 1e-8);
    EXPECT_EQ(ait.rowIndex(), bit.rowIndex());
    ++ait;
    ++bit;
  }
}

TEST(SparseBlas, setM) {
  smatrixd m(100, 100, 5 * 100);
  for (size_t r = 0; r < 100; r++)
    for (size_t c = 0; c < 5; c++)
      m.insert(r, c * 3, static_cast<double>(c));
  for (size_t r = 0; r < 100; r++) {
    EXPECT_EQ(m.valuesInRowCount(r), 5);
    m.iterateRow(r, [&](const double &v, size_t c) {
      ASSERT_NEAR(v, static_cast<double>(c / 3), 1e-8);
    });
  }
  SparseBlas2d::set(0.0, &m);
  for (size_t r = 0; r < 100; r++) {
    EXPECT_EQ(m.valuesInRowCount(r), 5);
    m.iterateRow(r, [&](const double &v, size_t c) {
      UNUSED_VARIABLE(c);
      ASSERT_NEAR(v, 0.0, 1e-8);
    });
  }
}

TEST(SparseBlas, setMM) {
  smatrixd m(100, 100, 5 * 100), m2(1, 1, 1);
  for (size_t r = 0; r < 100; r++)
    for (size_t c = 0; c < 5; c++)
      m.insert(r, c * 3, static_cast<double>(c));
  SparseBlas2d::set(m, &m2);
  smatrixd::const_iterator a(m), b(m2);
  while (a.next() && b.next()) {
    ASSERT_NEAR(a.value(), b.value(), 1e-8);
    EXPECT_EQ(a.rowIndex(), b.rowIndex());
    EXPECT_EQ(a.columnIndex(), b.columnIndex());
    ++a;
    ++b;
  }
}

TEST(SparseBlas, dot) {
  svecd A(15, 10), B(15, 10);
  double e = 0.0;
  for (int i = 0; i < 10; i++) {
    A.insert(i, i);
    B.insert(i + 5, i + 5);
    if (i >= 5 && i < 10)
      e += SQR(i);
  }
  double d = SparseBlas2d::dot(A, B);
  ASSERT_NEAR(d, e, 1e-6);
}

TEST(SparseBlas, aspy) {
  double a = 10.0;
  svecd x(200, 100), y(200, 200), z(200, 200);
  for (int i = 0; i < 200; i++) {
    if (i < 100)
      x.insert(i, i);
    y.insert(i, 5.0);
  }
  SparseBlas2d::axpy(a, x, y, &z);
  EXPECT_EQ(z.elementCount(), 200u);
  int count = 0;
  for (svecd::const_iterator it(z); it.next(); ++it) {
    count++;
    if (it.rowIndex() < 100)
      ASSERT_NEAR(it.value(), static_cast<double>(it.rowIndex() * 10 + 5),
                  1e-8);
    else
      ASSERT_NEAR(it.value(), 5.0, 1e-8);
  }
  EXPECT_EQ(count, 200);
}

TEST(SparseBlas, mvm) {
  smatrixd m(5, 5, 5 * 3);
  m.insert(0, 0, 1.0);
  m.insert(0, 2, 1.0);
  m.insert(0, 4, 1.0);
  m.insert(1, 0, 2.0);
  m.insert(1, 2, 2.0);
  m.insert(1, 3, 2.0);
  m.insert(2, 0, 3.0);
  m.insert(2, 1, 3.0);
  m.insert(2, 2, 3.0);
  m.insert(4, 0, 5.0);
  m.insert(4, 1, 5.0);
  m.insert(4, 2, 5.0);
  m.insert(4, 3, 5.0);
  m.insert(4, 4, 5.0);
  svecd v(5, 4), w(5, 0);
  v.insert(0, 10.0);
  v.insert(1, 20.0);
  v.insert(3, 40.0);
  v.insert(4, 50.0);
  double expected[5] = {1.0 * (10 + 0 + 50), 2.0 * (10 + 0 + 40),
                        3.0 * (10 + 20 + 0), 0.0, 5.0 * (10 + 20 + 40 + 50)};
  SparseBlas2d::mvm(m, v, &w);
  for (svecd::const_iterator it(w); it.next(); ++it) {
    static size_t count = 0;
    EXPECT_EQ(count, it.rowIndex());
    ASSERT_NEAR(it.value(), expected[count], 1e-8);
    count++;
    if (count == 3)
      count++;
  }
}

TEST(SparseBlas, residual) {
  smatrixd m(5, 5, 5 * 3);
  m.insert(0, 0, 1.0);
  m.insert(0, 2, 1.0);
  m.insert(0, 4, 1.0);
  m.insert(1, 0, 2.0);
  m.insert(1, 2, 2.0);
  m.insert(1, 3, 2.0);
  m.insert(2, 0, 3.0);
  m.insert(2, 1, 3.0);
  m.insert(2, 2, 3.0);
  m.insert(4, 0, 5.0);
  m.insert(4, 1, 5.0);
  m.insert(4, 2, 5.0);
  m.insert(4, 3, 5.0);
  m.insert(4, 4, 5.0);
  svecd v(5, 4);
  v.insert(0, 10.0);
  v.insert(1, 20.0);
  v.insert(3, 40.0);
  v.insert(4, 50.0);
  svecd b(5), w(5);
  SparseBlas2d::residual(m, v, b, &w);
  double expected[5] = {-1.0 * (10 + 0 + 50), -2.0 * (10 + 0 + 40),
                        -3.0 * (10 + 20 + 0), -0.0, -5.0 * (10 + 20 + 40 + 50)};
  EXPECT_EQ(w.elementCount(), 4u);
  size_t count = 0;
  for (svecd::const_iterator it(w); it.next(); ++it) {
    ASSERT_NEAR(it.value(), expected[count], 1e-8);
    count++;
    if (count == 3)
      count++;
  }
  EXPECT_EQ(count, 5u);
}

TEST(SparseBlas, l2Norm) {
  svecd v(100);
  double s = 0.0;
  for (int i = 0; i < 100; i += 2) {
    s += i * i;
    v.insert(i, i);
  }
  EXPECT_EQ(v.elementCount(), 50u);
  ASSERT_NEAR(sqrt(s), SparseBlas2d::l2Norm(v), 1e-8);
}
