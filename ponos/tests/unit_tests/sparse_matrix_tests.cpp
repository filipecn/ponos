#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;
//    0   1   2   3   4
// 0 3.0 0.0 1.0 2.0 0.0
// 1 0.0 4.0 0.0 0.0 0.0
// 2 0.0 7.0 5.0 9.0 0.0
// 3 0.0 0.0 0.0 0.0 0.0
// 4 0.0 0.0 0.0 6.0 5.0
TEST(SparseMatrixCRS, Iterate) {
  std::vector<double> ev[5];
  std::vector<size_t> ec[5];
  SparseMatrix<double> m(5, 5, 9);
  m.insert(0, 0, 3.0);
  ev[0].emplace_back(3.0);
  ec[0].emplace_back(0);
  m.insert(0, 2, 1.0);
  ev[0].emplace_back(1.0);
  ec[0].emplace_back(2);
  m.insert(0, 3, 2.0);
  ev[0].emplace_back(2.0);
  ec[0].emplace_back(3);
  m.insert(1, 1, 4.0);
  ev[1].emplace_back(4.0);
  ec[1].emplace_back(1);
  m.insert(2, 1, 7.0);
  ev[2].emplace_back(7.0);
  ec[2].emplace_back(1);
  m.insert(2, 2, 5.0);
  ev[2].emplace_back(5.0);
  ec[2].emplace_back(2);
  m.insert(2, 3, 9.0);
  ev[2].emplace_back(9.0);
  ec[2].emplace_back(3);
  m.insert(4, 3, 6.0);
  ev[4].emplace_back(6.0);
  ec[4].emplace_back(3);
  m.insert(4, 4, 5.0);
  ev[4].emplace_back(5.0);
  ec[4].emplace_back(4);
  EXPECT_EQ(m.valuesInRowCount(0), 3);
  EXPECT_EQ(m.valuesInRowCount(1), 1);
  EXPECT_EQ(m.valuesInRowCount(2), 3);
  EXPECT_EQ(m.valuesInRowCount(3), 0);
  EXPECT_EQ(m.valuesInRowCount(4), 2);
  double error = 1e-8;
  for (int i = 0; i < 5; i++) {
    std::vector<size_t> columns;
    std::vector<double> values;
    m.iterateRow(i, [&columns, &values](double &v, size_t j) {
      values.emplace_back(v);
      columns.emplace_back(j);
    });
    EXPECT_EQ(values.size(), ev[i].size());
    for (size_t j = 0; j < values.size(); j++)
      ASSERT_NEAR(ev[i][j], values[j], error);
    EXPECT_EQ(columns.size(), ec[i].size());
    for (size_t j = 0; j < columns.size(); j++)
      EXPECT_EQ(columns[j], ec[i][j]);
  }
  size_t expectedRowIndices[9] = {0, 0, 0, 1, 2, 2, 2, 4, 4};
  size_t k = 0;
  for (SparseMatrix<double>::const_iterator it(m); it.next(); ++it) {
    EXPECT_EQ(expectedRowIndices[k], it.rowIndex());
    k++;
  }
}
