#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;

TEST(TetrahedronMesh, init) {
  RawMesh rm;
  rm.vertexDescriptor.count = 5;
  rm.vertexDescriptor.elementSize = 3;
  rm.addVertex({1.0, 1.0, 0.0,   // 0
                1.0, 0.0, 1.0,   // 1
                2.0, 0.0, 0.0,   // 2
                0.0, 1.0, 0.0,   // 3
                0.0, 0.0, 0.0}); // 4
  rm.meshDescriptor.count = 2;
  rm.meshDescriptor.elementSize = 4;
  rm.addFace({{0, 0, 0},
              {1, 0, 0},
              {2, 0, 0},
              {4, 0, 0}, // 0
              {0, 0, 0},
              {1, 0, 0},
              {4, 0, 0},
              {3, 0, 0}}); // 1
  rm.primitiveType = GeometricPrimitiveType::TETRAHEDRA;
  TMesh<> tmesh(rm);
  // EXPECT_EQ(r, true);
}
// comentario do kennedy
