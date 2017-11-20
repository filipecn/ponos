#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>

using namespace ponos;

class TestBRep : public CppUnit::TestCase {
public:
  CPPUNIT_TEST_SUITE(TestBRep);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST(testFromRawMesh);
  CPPUNIT_TEST_SUITE_END();

  void test();
  void testFromRawMesh();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestBRep);

void TestBRep::test() {
  /*
   v0  11   v1  13  v2
    -----------------
    |\ 10   |\ 12   |
    | \  f1 | \  f3 |
   0|1 \   4|5 \   8|9
    |  2\3  |  6\7  |
    |f0  \  |f2  \  |
    |  14 \ |  16 \ |
    -----------------
   v3  15   v4 17   v5
  */
  HEMesh<float, 2> mesh;
  mesh.addVertex(Point<float, 2>({0, 1}));
  mesh.addVertex(Point<float, 2>({1, 1}));
  mesh.addVertex(Point<float, 2>({2, 1}));
  mesh.addVertex(Point<float, 2>({0, 0}));
  mesh.addVertex(Point<float, 2>({1, 0}));
  mesh.addVertex(Point<float, 2>({2, 0}));
  mesh.addEdge(0, 3);
  mesh.addEdge(4, 0);
  mesh.addEdge(4, 1);
  mesh.addEdge(5, 1);
  mesh.addEdge(5, 2);
  mesh.addEdge(1, 0);
  mesh.addEdge(2, 1);
  mesh.addEdge(3, 4);
  mesh.addEdge(4, 5);
  mesh.addFace({1, 14, 2});
  mesh.addFace({3, 4, 10});
  mesh.addFace({6, 5, 16});
  mesh.addFace({7, 8, 12});
  // mesh.traverseEdgesFromVertex(0, [](int e) { std::cout << e << " "; });
  std::cout << std::endl;
  mesh.traversePolygonEdges(0, [](int e) { std::cout << e << " "; });
  std::cout << std::endl;
  CPPUNIT_ASSERT(true);
}

void TestBRep::testFromRawMesh() {
  /*
   v0   9   v1  17  v2
    -----------------
    |\  8   |\  16  |
    | \  f1 | \  f3 |
   1|0 \   6|7 \  14|15
    |  4\5  | 12\13 |
    |f0  \  |f2  \  |
    |  2  \ |  10 \ |
    -----------------
   v3  3   v4  11    v5
  */
  RawMesh rm;
  rm.meshDescriptor.elementSize = 3;
  rm.meshDescriptor.count = 4;
  rm.vertexDescriptor.elementSize = 2;
  rm.vertexDescriptor.count = 6;
  rm.addVertex({0, 1});
  rm.addVertex({1, 1});
  rm.addVertex({2, 1});
  rm.addVertex({0, 0});
  rm.addVertex({1, 0});
  rm.addVertex({2, 0});
  rm.addFace({{0, 0, 0}, {3, 0, 0}, {4, 0, 0}});
  rm.addFace({{0, 0, 0}, {4, 0, 0}, {1, 0, 0}});
  rm.addFace({{1, 0, 0}, {4, 0, 0}, {5, 0, 0}});
  rm.addFace({{1, 0, 0}, {5, 0, 0}, {2, 0, 0}});
  HEMesh<float, 2> mesh(&rm);
  const std::vector<HEMesh<float, 2>::Edge> &edges = mesh.getEdges();
  CPPUNIT_ASSERT(edges.size() == 18);
  CPPUNIT_ASSERT(edges[0].orig == 0 && edges[0].dest == 3);
  CPPUNIT_ASSERT(edges[1].orig == 3 && edges[1].dest == 0);
  CPPUNIT_ASSERT(edges[2].orig == 3 && edges[2].dest == 4);
  CPPUNIT_ASSERT(edges[3].orig == 4 && edges[3].dest == 3);
  CPPUNIT_ASSERT(edges[4].orig == 4 && edges[4].dest == 0);
  CPPUNIT_ASSERT(edges[5].orig == 0 && edges[5].dest == 4);
  CPPUNIT_ASSERT(edges[6].orig == 4 && edges[6].dest == 1);
  CPPUNIT_ASSERT(edges[7].orig == 1 && edges[7].dest == 4);
  CPPUNIT_ASSERT(edges[8].orig == 1 && edges[8].dest == 0);
  CPPUNIT_ASSERT(edges[9].orig == 0 && edges[9].dest == 1);
  CPPUNIT_ASSERT(edges[10].orig == 4 && edges[10].dest == 5);
  CPPUNIT_ASSERT(edges[11].orig == 5 && edges[11].dest == 4);
  CPPUNIT_ASSERT(edges[12].orig == 5 && edges[12].dest == 1);
  CPPUNIT_ASSERT(edges[13].orig == 1 && edges[13].dest == 5);
  CPPUNIT_ASSERT(edges[14].orig == 5 && edges[14].dest == 2);
  CPPUNIT_ASSERT(edges[15].orig == 2 && edges[15].dest == 5);
  CPPUNIT_ASSERT(edges[16].orig == 2 && edges[16].dest == 1);
  CPPUNIT_ASSERT(edges[17].orig == 1 && edges[17].dest == 2);
  int fe[3];
  mesh.traversePolygonEdges(0, [&fe](int e) {
    static int i = 0;
    fe[i++] = e;
  });
  CPPUNIT_ASSERT(fe[0] == 4 && fe[1] == 0 && fe[2] == 2);
  mesh.traversePolygonEdges(1, [&fe](int e) {
    static int i = 0;
    fe[i++] = e;
  });
  CPPUNIT_ASSERT(fe[0] == 8 && fe[1] == 5 && fe[2] == 6);
  mesh.traversePolygonEdges(2, [&fe](int e) {
    static int i = 0;
    fe[i++] = e;
  });
  CPPUNIT_ASSERT(fe[0] == 12 && fe[1] == 7 && fe[2] == 10);
  mesh.traversePolygonEdges(3, [&fe](int e) {
    static int i = 0;
    fe[i++] = e;
  });
  CPPUNIT_ASSERT(fe[0] == 16 && fe[1] == 13 && fe[2] == 14);
  CPPUNIT_ASSERT(edges[0].next == 2);
  CPPUNIT_ASSERT(edges[2].next == 4);
  CPPUNIT_ASSERT(edges[4].next == 0);
  CPPUNIT_ASSERT(edges[5].next == 6);
  CPPUNIT_ASSERT(edges[6].next == 8);
  CPPUNIT_ASSERT(edges[8].next == 5);
  CPPUNIT_ASSERT(edges[7].next == 10);
  CPPUNIT_ASSERT(edges[10].next == 12);
  CPPUNIT_ASSERT(edges[12].next == 7);
  CPPUNIT_ASSERT(edges[13].next == 14);
  CPPUNIT_ASSERT(edges[14].next == 16);
  CPPUNIT_ASSERT(edges[16].next == 13);
  std::cout << edges[1].next << std::endl;
  std::cout << edges[3].next << std::endl;
  std::cout << edges[11].next << std::endl;
  std::cout << edges[15].next << std::endl;
  std::cout << edges[17].next << std::endl;
  std::cout << edges[9].next << std::endl;
  // mesh.traverseEdgesFromVertex(0, [](int e) { std::cout << e << " "; });
  // std::cout << std::endl;
}
