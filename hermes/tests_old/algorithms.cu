#include <gtest/gtest.h>
#include <hermes/hermes.h>
#include <thrust/device_vector.h>

using namespace hermes::cuda;

TEST(Algorithms, MarchingSquares) {
  vec2u size(6);
  RegularGrid2Hf grid(size);
  grid.setSpacing(vec2f(0.2));
  auto acc = grid.accessor();
  for (size_t j = 0; j < size.y; j++)
    for (size_t i = 0; i < size.x; i++)
      acc(i, j) = (acc.worldPosition(i, j) - point2f(0.5f)).length() - 0.2f;
  RegularGrid2Df d_grid(size);
  d_grid.setSpacing(vec2f(0.2f));
  memcpy(d_grid.data(), grid.data());
  MemoryBlock1Df vertices;
  MemoryBlock1Du indices;
  MarchingSquares::extractIsoline(d_grid, vertices, indices);
  EXPECT_EQ(vertices.size(), 32u);
  MemoryBlock1Hf h_vertices(vertices.size());
  h_vertices.allocate();
  memcpy(h_vertices, vertices);
  auto hAcc = h_vertices.accessor();
  for (size_t i = 0; i < vertices.size() / 2; i++)
    EXPECT_NEAR(distance(point2f(hAcc[i * 2], hAcc[i * 2 + 1]), point2f(0.5f)),
                0.2f, 1e-1f);
}

TEST(Algorithms, MarchingCubes) {
  vec3u size(10);
  vec3f spacing(0.1);
  point3f center(0.5);
  float r = 0.3f;
  RegularGrid3Hf grid(size);
  grid.setSpacing(spacing);
  auto acc = grid.accessor();
  for (auto e : acc)
    e.value = (e.worldPosition() - center).length() - r;
  RegularGrid3Df d_grid(size);
  d_grid.setSpacing(spacing);
  memcpy(d_grid.data(), grid.data());
  MemoryBlock1Df normals;
  MemoryBlock1Df vertices;
  MemoryBlock1Du indices;
  MarchingCubes::extractSurface(d_grid, vertices, indices, 0.f, &normals);
  MemoryBlock1Hf h_vertices(vertices.size());
  h_vertices.allocate();
  MemoryBlock1Hf h_normals(normals.size());
  h_normals.allocate();
  memcpy(h_vertices, vertices);
  memcpy(h_normals, normals);
  auto hAcc = h_vertices.accessor();
  auto nAcc = h_normals.accessor();
  for (size_t i = 0; i < vertices.size() / 3; i++) {
    point3f pos = point3f(hAcc[i * 3], hAcc[i * 3 + 1], hAcc[i * 3 + 2]);
    EXPECT_NEAR(distance(pos, center), r, 1e-3f);
    vec3f npos = normalize(pos - center);
    for (int j = 0; j < 3; j++)
      EXPECT_NEAR(npos[j], nAcc[i * 3 + j], 1e-1f);
  }
}