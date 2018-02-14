#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;

TEST(Octree, Constructors) {
  {
    Octree<float> tree;
    EXPECT_EQ(tree.height(), 0u);
    EXPECT_EQ(tree.nodeCount(), 0u);
  }
  {
    Octree<float> tree(BBox::unitBox(), 4);
    EXPECT_EQ(tree.height(), 4u);
    EXPECT_EQ(tree.nodeCount(), 1u + 8u + 8u * 8u + 8u * 8u * 8u + 8u * 8u * 8u * 8u);
  }
  {
    Octree<float> tree(BBox::unitBox(), [](Octree<float>::Node &node) {
      UNUSED_VARIABLE(node);
      return false;
    });
    EXPECT_EQ(tree.height(), 0u);
    EXPECT_EQ(tree.nodeCount(), 1u);
  }
  {
    Octree<float> tree(BBox::unitBox(), [](Octree<float>::Node &node) {
      return node.level() < 1;
    });
    EXPECT_EQ(tree.height(), 1u);
    EXPECT_EQ(tree.nodeCount(), 9u);
  }
  {
    Octree<float> tree(BBox::unitBox(), [](Octree<float>::Node &node) {
      return node.level() < 4;
    });
    EXPECT_EQ(tree.height(), 4u);
    EXPECT_EQ(tree.nodeCount(), 1u + 8u + 8u * 8u + 8u * 8u * 8u + 8u * 8u * 8u * 8u);
  }
  {
    Octree<float> tree(
        BBox::unitBox(), 1.f,
        [](Octree<float>::Node &node) { return node.level() < 1; },
        [](Octree<float>::Node &node) {
          if (node.id() > 0u)
            EXPECT_TRUE(node.isLeaf());
          else
            EXPECT_TRUE(!node.isLeaf());
        });
  }
}

TEST(Octree, Traversal) {
  {
    Octree<float> tree(BBox::unitBox(), 1);
    uint count = 0;
    tree.traverse([&](Octree<float>::Node& node){
      node.data = 123.f;
      EXPECT_EQ(node.id(), count++);
      return true;
    });
    EXPECT_EQ(count, 9u);
    count = 0;
    std::vector<BBox> regions = {
        BBox(Point3(0,0,0), Point3(1,1,1)),
        BBox(Point3(0.0,0.0,0.0), Point3(0.5,0.5,0.5)),
        BBox(Point3(0.5,0.0,0.0), Point3(1.0,0.5,0.5)),
        BBox(Point3(0.0,0.5,0.0), Point3(0.5,1.0,0.5)),
        BBox(Point3(0.5,0.5,0.0), Point3(1.0,1.0,0.5)),
        BBox(Point3(0.0,0.0,0.5), Point3(0.5,0.5,1.0)),
        BBox(Point3(0.5,0.0,0.5), Point3(1.0,0.5,1.0)),
        BBox(Point3(0.0,0.5,0.5), Point3(0.5,1.0,1.0)),
        BBox(Point3(0.5,0.5,0.5), Point3(1.0,1.0,1.0))
    };
    tree.traverse([&](const Octree<float>::Node& node){
      EXPECT_FLOAT_EQ(node.data, 123.f);
      EXPECT_EQ(node.id(), count);
      // check region
      auto r = node.region();
      EXPECT_EQ(r.pMin, regions[count].pMin);
      EXPECT_EQ(r.pMax, regions[count].pMax);
      float s = node.region().size(0);
      EXPECT_FLOAT_EQ(node.region().size(1), s);
      EXPECT_FLOAT_EQ(node.region().size(2), s);
      float es = 1. / pow(2., 1. * node.level());
      EXPECT_FLOAT_EQ(s, es);
      s = node.region().size(1);
      EXPECT_FLOAT_EQ(s, es);
      count++;
      return true;
    });
    EXPECT_EQ(count, 9u);
  }
}