#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;

TEST(TetrahedronMesh, simple) {
  RawMesh rm;
  rm.positionDescriptor.count = 5;
  rm.positionDescriptor.elementSize = 3;
  rm.addPosition({1.0, 1.0, 0.0,   // 0
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
  EXPECT_EQ(tmesh.tetrahedron.size(), 2u);
  EXPECT_EQ(tmesh.faces.size(), 8u);
  EXPECT_EQ(tmesh.edges.size(), 9u);
  EXPECT_EQ(tmesh.vertices.size(), 5u);
  {
    auto v = tmesh.tetrahedronVertices(0);
    EXPECT_EQ(v[0], 0u);
    EXPECT_EQ(v[1], 1u);
    EXPECT_EQ(v[2], 2u);
    EXPECT_EQ(v[3], 4u);
    auto f = tmesh.tetrahedronFaces(0);
    EXPECT_EQ(f[0], 3u);
    EXPECT_EQ(f[1], 1u);
    EXPECT_EQ(f[2], 2u);
    EXPECT_EQ(f[3], 0u);
    EXPECT_EQ(tmesh.faces[f[0]].hface, 4);
    auto t = tmesh.tetrahedronNeighbours(0);
    EXPECT_EQ(t.size(), 1u);
    EXPECT_EQ(t[0], 1u);
  }
  {
    auto v = tmesh.tetrahedronVertices(1);
    EXPECT_EQ(v[0], 0u);
    EXPECT_EQ(v[1], 1u);
    EXPECT_EQ(v[2], 3u);
    EXPECT_EQ(v[3], 4u);
    auto f = tmesh.tetrahedronFaces(1);
    EXPECT_EQ(f[0], 7u);
    EXPECT_EQ(f[1], 5u);
    EXPECT_EQ(f[2], 6u);
    EXPECT_EQ(f[3], 4u);
    EXPECT_EQ(tmesh.faces[f[3]].hface, 3);
    auto t = tmesh.tetrahedronNeighbours(1);
    EXPECT_EQ(t.size(), 1u);
    EXPECT_EQ(t[0], 0u);
  }
  auto vn = tmesh.vertexNeighbours(0);
  EXPECT_EQ(vn.size(), 4u);
  EXPECT_EQ(vn[0], 1u);
  EXPECT_EQ(vn[1], 2u);
  EXPECT_EQ(vn[2], 4u);
  EXPECT_EQ(vn[3], 3u);
  auto tn = tmesh.vertexTetrahedra(0);
  EXPECT_EQ(tn.size(), 2u);
  EXPECT_EQ(tn[0], 0u);
  EXPECT_EQ(tn[1], 1u);
  vn = tmesh.vertexNeighbours(1);
  EXPECT_EQ(vn.size(), 4u);
  EXPECT_EQ(vn[0], 0u);
  EXPECT_EQ(vn[1], 2u);
  EXPECT_EQ(vn[2], 4u);
  EXPECT_EQ(vn[3], 3u);
  tn = tmesh.vertexTetrahedra(1);
  EXPECT_EQ(tn.size(), 2u);
  EXPECT_EQ(tn[0], 0u);
  EXPECT_EQ(tn[1], 1u);
  vn = tmesh.vertexNeighbours(2);
  EXPECT_EQ(vn.size(), 3u);
  EXPECT_EQ(vn[0], 0u);
  EXPECT_EQ(vn[1], 1u);
  EXPECT_EQ(vn[2], 4u);
  tn = tmesh.vertexTetrahedra(2);
  EXPECT_EQ(tn.size(), 1u);
  EXPECT_EQ(tn[0], 0u);
  vn = tmesh.vertexNeighbours(3);
  EXPECT_EQ(vn.size(), 3u);
  EXPECT_EQ(vn[0], 0u);
  EXPECT_EQ(vn[1], 1u);
  EXPECT_EQ(vn[2], 4u);
  tn = tmesh.vertexTetrahedra(3);
  EXPECT_EQ(tn.size(), 1u);
  EXPECT_EQ(tn[0], 1u);
  vn = tmesh.vertexNeighbours(4);
  EXPECT_EQ(vn.size(), 4u);
  EXPECT_EQ(vn[0], 0u);
  EXPECT_EQ(vn[1], 1u);
  EXPECT_EQ(vn[2], 2u);
  EXPECT_EQ(vn[3], 3u);
  tn = tmesh.vertexTetrahedra(4);
  EXPECT_EQ(tn.size(), 2u);
  EXPECT_EQ(tn[0], 0u);
  EXPECT_EQ(tn[1], 1u);
}

TEST(TetrahedronMesh, tetrahedralizeTest) {
  auto surface = ponos::create_icosphere_mesh(ponos::Point3(), 1.f, 0, false, false);
  RawMesh tetmesh;
  tetrahedralize(surface, &tetmesh);
  delete surface;
  TMesh<> tmesh(tetmesh);
  EXPECT_EQ(tmesh.tetrahedron.size(), 15u);
  EXPECT_EQ(tmesh.faces.size(), 60u);
  EXPECT_EQ(tmesh.edges.size(), 36u);
  EXPECT_EQ(tmesh.vertices.size(), 12u);
}

TEST(TetrahedronMesh, fastMarch) {
  {
    RawMesh rm;
    rm.positionDescriptor.count = 7;
    rm.positionDescriptor.elementSize = 3;
    rm.addPosition({-1.f, 0.0, 1.f,   // 0
                    1.0, 0.0, 0.0,   // 1
                    0.0, 1.0, 0.0,   // 2
                    0.0, -1.f, 0.0,   // 3
                    -1.f, 0.0, -1.f,   // 4
                    1.0, 0.0, 2.0, // 5
                    1.0, -1.f, -1.f // 6
                   });
    rm.meshDescriptor.count = 4;
    rm.meshDescriptor.elementSize = 4;
    rm.addFace({0, 1, 2, 4, // 0
                0, 1, 3, 4, // 1
                0, 1, 2, 5, // 2
                0, 1, 5, 6 // 3
               }); 
    rm.primitiveType = GeometricPrimitiveType::TETRAHEDRA;
    TMesh<> tmesh(rm);
    fastMarchTetraheda(&tmesh, {3}, &tmesh);
    for (auto v : tmesh.vertices)
      std::cerr << v.data << " ";
    std::cerr << std::endl;
  }
  {
    RawMesh rm;
    rm.positionDescriptor.count = 9;
    rm.positionDescriptor.elementSize = 3;
    rm.addPosition({0.0, 0.0, 0.0,   // 0
                    1.0, 0.0, 0.0,   // 1
                    0.0, 1.0, 0.0,   // 2
                    1.0, 1.0, 0.0,   // 3
                    0.0, 0.0, 1.0,   // 4
                    1.0, 0.0, 1.0,   // 5
                    0.0, 1.0, 1.0,   // 6
                    1.0, 1.0, 1.0,   // 7
                    0.5, 0.5, 0.5}); // 8
    rm.meshDescriptor.count = 12;
    rm.meshDescriptor.elementSize = 4;
    rm.addFace({0, 1, 2, 8, // 0
                2, 1, 3, 8, // 1
                0, 4, 6, 8, // 2
                0, 6, 2, 8, // 3
                4, 5, 6, 8, // 4
                6, 5, 7, 8, // 5
                1, 5, 7, 8, // 6
                1, 7, 3, 8, // 7
                2, 3, 6, 8, // 8
                3, 6, 7, 8, // 9
                1, 4, 5, 8, //10
                0, 1, 4, 8, //11
               }); // 1
    rm.primitiveType = GeometricPrimitiveType::TETRAHEDRA;
    TMesh<> tmesh(rm);
    fastMarchTetraheda(&tmesh, {0}, &tmesh);
    for (auto v : tmesh.vertices)
      std::cerr << v.data << " ";
    std::cerr << std::endl;
  }
}