#include <gtest/gtest.h>
#include <ponos.h>

using namespace ponos;

template <typename T> class CustomVector : public VectorInterface<T> {
public:
  CustomVector(std::vector<T> _v) : v(_v) {}
  T operator[](size_t i) const override { return v[i]; }
  T &operator[](size_t i) override { return v[i]; }
  size_t size() const override { return v.size(); }
  std::vector<T> v;
};

template <int D, typename T> class CustomMatrix : public MatrixInterface<T> {
public:
  CustomMatrix() {
    for (int i = 0; i < D; i++)
      for (int j = 0; j < D; j++)
        m[i][j] = 0.f;
  }
  CustomMatrix(std::vector<T> v) {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        m[i][j] = v[i * 4 + j];
  }
  T operator()(size_t i, size_t j) const override { return m[i][j]; }
  T &operator()(size_t i, size_t j) override { return m[i][j]; }
  void iterateRow(size_t i,
                  std::function<void(const T &, size_t)> f) const override {
    for (size_t j = 0; j < D; j++)
      f(m[i][j], j);
  }
  virtual void iterateRow(size_t i,
                          std::function<void(T &, size_t)> f) override {
    exit(1);
  }
  virtual void
  iterateColumn(size_t j,
                std::function<void(const T &, size_t)> f) const override {
    for (size_t i = 0; i < D; i++)
      f(m[i][j], i);
  }
  virtual void iterateColumn(size_t j,
                             std::function<void(T &, size_t)> f) override {}
  size_t rowCount() const override { return D; }
  size_t columnCount() const override { return D; }

  T m[D][D];
};

TEST(CustomMatrix, ConstAccessor) {
  CustomMatrix<4, int> m(linspace(1, 16, 16));
  MatrixConstAccessor<CustomMatrix<4, int>, int> acc(&m);
  for (size_t i = 0; i < 4; i++)
    acc.iterateRow(i, [](const int &v, size_t j) {
      static int k = 0;
      EXPECT_EQ(k, v);
      k++;
    });
  MatrixConstAccessor<CustomMatrix<4, int>, int> acct(&m, true);
  for (size_t i = 0; i < 4; i++) {
    int k = i;
    acct.iterateRow(i, [&k](const int &v, size_t j) {
      EXPECT_EQ(k, v);
      k += 4;
    });
  }
}

TEST(CustomMatrix, MVM) {
  CustomMatrix<4, int> m(linspace(1, 16, 16));
  CustomVector<int> v(linspace(1, 4, 4));
  CustomVector<int> r(linspace(1, 4, 4));
  MatrixConstAccessor<CustomMatrix<4, int>, int> acc(&m);
  mvm(acc, &v, &r);
  for (size_t i = 0; i < 4; i++) {
    int s = 0;
    for (size_t j = 0; j < 4; j++)
      s += m(i, j) * v[j];
    EXPECT_EQ(s, r[i]);
  }
  MatrixConstAccessor<CustomMatrix<4, int>, int> acct(&m, true);
  mvm(acct, &v, &r);
  for (size_t i = 0; i < 4; i++) {
    int s = 0;
    for (size_t j = 0; j < 4; j++)
      s += m(j, i) * v[j];
    EXPECT_EQ(s, r[i]);
  }
}

TEST(CustomMatrix, axpy) {
  CustomVector<int> x(linspace(1, 16, 16));
  CustomVector<int> y(linspace(1, 16, 16));
  CustomVector<int> r(linspace(1, 16, 16));
  axpy(-2, &x, &y, &r);
  for (size_t i = 0; i < 16; i++)
    EXPECT_EQ(r[i], -2 * x[i] + y[i]);
}

TEST(CustomMatrix, residual) {
  CustomMatrix<4, int> m(linspace(1, 16, 16));
  CustomVector<int> x(linspace(1, 4, 4));
  CustomVector<int> b(linspace(1, 4, 4));
  CustomVector<int> r(linspace(1, 4, 4));
  MatrixConstAccessor<CustomMatrix<4, int>, int> acc(&m);
  residual(acc, &x, &b, &r);
  for (size_t i = 0; i < 4; i++) {
    int s = 0;
    for (size_t j = 0; j < 4; j++)
      s += m(i, j) * x[j];
    EXPECT_EQ(r[i], b[i] - s);
  }
  MatrixConstAccessor<CustomMatrix<4, int>, int> acct(&m, true);
  residual(acct, &x, &b, &r);
  for (size_t i = 0; i < 4; i++) {
    int s = 0;
    for (size_t j = 0; j < 4; j++)
      s += m(j, i) * x[j];
    EXPECT_EQ(r[i], b[i] - s);
  }
}

TEST(CustomMatrix, norm) {
  CustomVector<int> x(linspace(1, 16, 16));
  double s = norm(&x);
  double sum = 0;
  for (size_t i = 0; i < x.size(); i++)
    sum += x[i] * x[i];
  ASSERT_NEAR(s, sqrt(sum), 1e-8);
}

TEST(CustomMatrix, infnorm) {
  CustomVector<int> x(linspace(1, 16, 16));
  x[10] = -10;
  int s = infnorm(&x);
  int sum = 0;
  for (size_t i = 0; i < x.size(); i++)
    sum = std::max(sum, static_cast<int>(abs(x[i])));
  EXPECT_EQ(sum, s);
}

TEST(CustomMatrix, dSolve) {
  CustomMatrix<4, int> m(linspace(1, 16, 16));
  CustomVector<int> b(linspace(2, 5, 4));
  CustomVector<int> x(linspace(1, 4, 4));
  MatrixConstAccessor<CustomMatrix<4, int>, int> acc(&m);
  dSolve(acc, &b, &x);
  for (size_t i = 0; i < x.size(); i++)
    ASSERT_NEAR(x[i], ((m(i, i) != 0) ? b[i] / m(i, i) : b[i]), 1e-8);
}

TEST(CustomMatrix, dot) {
  CustomVector<int> b(linspace(2, 5, 4));
  CustomVector<int> x(linspace(1, 4, 4));
  int d = dot(&b, &x);
  int sum = 0;
  for (size_t i = 0; i < x.size(); i++)
    sum += x[i] * b[i];
  EXPECT_EQ(d, sum);
}

TEST(CustomMatrix, bicg_simple) {
  CustomMatrix<2, float> m;
  m(0, 0) = 4;
  m(0, 1) = 1;
  m(1, 0) = 1;
  m(1, 1) = 3;
  CustomVector<float> b(std::vector<float>(2, 0.f));
  b[0] = 1;
  b[1] = 2;
  {
    MatrixConstAccessor<CustomMatrix<2, float>, float> A(&m);
    MatrixConstAccessor<CustomMatrix<2, float>, float> At(&m, true);
    CustomVector<float> x(std::vector<float>(2, 0.f));
    CustomVector<float> r(std::vector<float>(2, 0.f));
    CustomVector<float> rt(std::vector<float>(2, 0.f));
    CustomVector<float> p(std::vector<float>(2, 0.f));
    CustomVector<float> pt(std::vector<float>(2, 0.f));
    CustomVector<float> z(std::vector<float>(2, 0.f));
    CustomVector<float> zt(std::vector<float>(2, 0.f));
    bicg(A, At, &x, &b, &r, &rt, &p, &pt, &z, &zt, 3, 1e-8, 1);
    for (size_t i = 0; i < 2; i++)
      std::cout << x[i] << " ";
    std::cout << std::endl;
  }
}

TEST(CustomMatrix, bicg) {
  CustomMatrix<8, float> m;
  m(0, 0) = 3.100;
  m(0, 1) = -1.000;
  m(0, 2) = -1.000;
  m(0, 5) = -1.000;
  m(1, 0) = -1.000;
  m(1, 1) = 4.000;
  m(1, 2) = -1.000;
  m(1, 4) = -1.000;
  m(1, 6) = -1.000;
  m(2, 0) = -1.000;
  m(2, 1) = -1.000;
  m(2, 2) = 5.000;
  m(2, 3) = -1.000;
  m(2, 5) = -1.000;
  m(2, 7) = -1.000;
  m(3, 2) = -1.000;
  m(3, 3) = 4.100;
  m(3, 4) = -1.000;
  m(3, 5) = -1.000;
  m(3, 6) = -1.000;
  m(4, 1) = -1.000;
  m(4, 3) = -1.000;
  m(4, 4) = 4.500;
  m(4, 5) = -1.000;
  m(4, 7) = -1.000;
  m(5, 0) = -1.000;
  m(5, 2) = -1.000;
  m(5, 3) = -1.000;
  m(5, 4) = -1.000;
  m(5, 5) = 5.000;
  m(5, 6) = -1.000;
  m(6, 1) = -1.000;
  m(6, 3) = -1.000;
  m(6, 5) = -1.000;
  m(6, 6) = 3.500;
  m(6, 7) = -1.000;
  m(7, 2) = -1.000;
  m(7, 4) = -1.000;
  m(7, 6) = -1.000;
  m(7, 7) = 3.300;
  std::vector<float> b(8, 0.f);
  b[0] = 0.100;
  b[3] = 0.100;
  b[4] = 0.500;
  b[6] = -0.500;
  b[7] = 0.300;
  CustomVector<float> B(b);
  {
    MatrixConstAccessor<CustomMatrix<8, float>, float> A(&m);
    MatrixConstAccessor<CustomMatrix<8, float>, float> At(&m, true);
    CustomVector<float> x(std::vector<float>(8, 0.f));
    CustomVector<float> r(std::vector<float>(8, 0.f));
    CustomVector<float> rt(std::vector<float>(8, 0.f));
    CustomVector<float> p(std::vector<float>(8, 0.f));
    CustomVector<float> pt(std::vector<float>(8, 0.f));
    CustomVector<float> z(std::vector<float>(8, 0.f));
    CustomVector<float> zt(std::vector<float>(8, 0.f));
    bicg(A, At, &x, &B, &r, &rt, &p, &pt, &z, &zt, 10, 1e-8, 1);
    for (size_t i = 0; i < 8; i++)
      std::cout << x[i] << " ";
    std::cout << std::endl;
  }
}
