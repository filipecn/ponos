#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;

TEST(ZPointSet, Constructors) {
  {
    ZPointSet s;
    EXPECT_EQ(s.size(), 0u);
  }
  {
    ZPointSet s(4);
    EXPECT_EQ(s.size(), 0u);
  }
}

TEST(ZPointSet, Iterator) {
  {
    ZPointSet z(2u);
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
          z.add(Point3(1.f * i, 1.f * j, 1.f * k));
    ZPointSet::iterator it(z);
    EXPECT_EQ(it.count(), 2u * 2u * 2u);
    int i = 0, j = 0, k = 0;
    uint count = 0;
    while (it.next()) {
      EXPECT_EQ(Point3(1. * i, 1. * j, 1. * k), it.getWorldPosition());
      EXPECT_EQ(count++, it.getId());
      k++;
      if (k >= 2) {
        k = 0;
        j++;
        if (j >= 2) {
          j = 0;
          i++;
        }
      }
      auto wp = it.getWorldPosition();
      EXPECT_EQ(
          it.pointElement()->zcode,
          mortonCode(wp.x, wp.y, wp.z));
      ++it;
    }
    EXPECT_EQ(count, z.size());
  }
  {
    ZPointSet s(4);
    //                           z y x z y x    m
    s.add(Point3(1, 1, 1)); // 0   0 0 0 1 1 1    7
    s.add(Point3(3, 1, 1)); // 1   0 0 1 1 1 1   15
    s.add(Point3(1, 1, 3)); // 2   1 0 0 1 1 1   39
    s.add(Point3(3, 1, 3)); // 3   1 0 1 1 1 1   47
    s.add(Point3(1, 3, 3)); // 4   1 1 0 1 1 1   55
    s.add(Point3(1, 3, 1)); // 5   0 1 0 1 1 1   23
    s.add(Point3(2, 2, 1)); // 6   0 1 1 1 0 0   28
    s.add(Point3(3, 2, 1)); // 7   0 1 1 1 0 1   29
    s.add(Point3(2, 3, 1)); // 8   0 1 1 1 1 0   30
    s.add(Point3(3, 3, 1)); // 9   0 1 1 1 1 1   31
    s.add(Point3(2, 2, 0)); // 10  0 1 1 0 0 0   24
    s.add(Point3(3, 2, 0)); // 11  0 1 1 0 0 1   25
    s.add(Point3(2, 3, 0)); // 12  0 1 1 0 1 0   26
    s.add(Point3(3, 3, 0)); // 13  0 1 1 0 1 1   27
    s.add(Point3(3, 3, 3)); // 14  1 1 1 1 1 1   63
    s.update();
    s.update();
    {
      ZPointSet::iterator it(s);
      EXPECT_EQ(it.count(), 15u);
    }
    {
      ZPointSet::iterator it(s, 55);
      EXPECT_EQ(it.count(), 2u);
    }
    {
      ZPointSet::iterator it(s, 24);
      EXPECT_EQ(it.count(), 12u);
    }
    {
      ZPointSet::iterator it(s, 24, 1);
      EXPECT_EQ(it.count(), 8u);
      std::vector<uint> expectedIds = {10, 11, 12, 13, 6, 7, 8, 9};
      int k = 0;
      for (; it.next(); ++it)
        EXPECT_EQ(it.getId(), expectedIds[k++]);
    }
  }
}

TEST(ZPointSet, Update) {
  {
    ZPointSet s(4);
    s.add(Point3(1, 1, 1));
    s.add(Point3(0, 1, 1));
    s.add(Point3(1, 0, 1));
    s.add(Point3(0, 0, 1));
    s.add(Point3(1, 1, 0));
    s.add(Point3(0, 1, 0));
    s.add(Point3(1, 0, 0));
    s.add(Point3(0, 0, 0));
    uint k = 0;
    for (ZPointSet::iterator it(s); it.next(); ++it)
      EXPECT_EQ(it.getId(), k++);
    s.update();
    k--;
    for (ZPointSet::iterator it(s); it.next(); ++it)
      EXPECT_EQ(it.getId(), k--);
  }
}

TEST(ZPointSet, SearchTree) {
  {
    ZPointSet s(2);
    ZPointSet::search_tree tree(s);
    EXPECT_EQ(tree.height(), 1u);
    uint zcode = 0;
    tree.traverse([&](const Octree<ZPointSet::NodeElement>::Node &node) -> bool {
      if (node.level() > 0)
        EXPECT_EQ(node.data.zcode, zcode++);
      else
        EXPECT_EQ(node.data.zcode, 0u);
      return true;
    });
  }
  {
    ZPointSet s(4);
    ZPointSet::search_tree tree(s);
    EXPECT_EQ(tree.height(), 2u);
    uint zcode = 0;
    tree.traverse([&](const Octree<ZPointSet::NodeElement>::Node &node) -> bool {
      if (node.isLeaf()) {
        EXPECT_EQ(node.data.zcode, zcode++);
      }
      return true;
    });
  }
  {
    std::vector<Point3> points;
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        for (int k = 0; k < 2; ++k)
          points.emplace_back(i, j, k);
    ZPointSet s(2);
    for (auto p : points)
      s.add(p);
    //s.update();
    ZPointSet::search_tree tree(s);
    int count = 0;
    tree.iteratePoints(BBox3(Point3(0, 0, 0), Point3(2, 2, 2)), [&](uint id) {
      UNUSED_VARIABLE(id);
      count++;
    });
    EXPECT_EQ(count, 2 * 2 * 2);
    count = 0;
    tree.iteratePoints(BBox3(Point3(1, 0, 0), Point3(2, 2, 2)), [&](uint id) {
      UNUSED_VARIABLE(id);
      count++;
    });
    EXPECT_EQ(count, 2 * 2);
  }
  {
    ZPointSet z(4);
    BBox3 region(Point3(0, 0, 0), Point3(4, 4, 4));
    std::vector<Point3> points;
    RNGSampler rng(new HaltonSequence(3), new HaltonSequence(5), new HaltonSequence(7));
    for (int i = 0; i < 10000; i++) {
      points.emplace_back(rng.sample(region));
      z.add(points[points.size() - 1]);
    }
    z.update();
    std::set<uint> s;
    ZPointSet::search_tree tree(z);
    tree.traverse([&](const Octree<ZPointSet::NodeElement>::Node &node) {
      BBox3 r = node.region();
      std::vector<uint> result;
      tree.iteratePoints(r, [&](uint id) { result.emplace_back(id); });
      std::sort(result.begin(), result.end());
      std::vector<uint> bfResult;
      for (uint k = 0; k < points.size(); k++)
        if (r.contains(points[k]))
          bfResult.emplace_back(k);
      std::sort(bfResult.begin(), bfResult.end());
      EXPECT_EQ(result.size(), bfResult.size());
      for (uint k = 0; k < result.size(); k++) {
        EXPECT_EQ(result[k], bfResult[k]);
        s.insert(result[k]);
      }
      return true;
    });
    EXPECT_EQ(s.size(), points.size());
  }
  { // TEST COMPUTE POINTS INDICES
    std::vector<Point3> points;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 4; ++k)
          points.emplace_back(i, j, k);
    ZPointSet s(4);
    for (auto p : points)
      s.add(p);
    s.update();
    ZPointSet::search_tree tree(s);
    tree.computePointsIndices();
  }
}

TEST(ZPointSet, Search) {
  {
    ZPointSet z(16);
    BBox3 region(Point3(0, 0, 0), Point3(16, 16, 16));
    std::vector<Point3> points;
    RNGSampler rng(new HaltonSequence(3), new HaltonSequence(5), new HaltonSequence(7));
    for (int i = 0; i < 1000; i++) {
      points.emplace_back(rng.sample(region));
      z.add(points[points.size() - 1]);
    }
    region.expand(4);
    for (int i = 0; i < 1000; i++) {
      // generate a bounding box
      BBox3 r(rng.sample(region), rng.sample(region));
      std::vector<uint> result;
      z.search(r, [&](uint id) { result.emplace_back(id); });
      std::sort(result.begin(), result.end());
      std::vector<uint> bfResult;
      for (uint k = 0; k < points.size(); k++)
        if (r.contains(points[k]))
          bfResult.emplace_back(k);
      std::sort(bfResult.begin(), bfResult.end());
      EXPECT_EQ(result.size(), bfResult.size());
      for (uint k = 0; k < result.size(); k++)
        EXPECT_EQ(result[k], bfResult[k]);
    }
  }
  {
    ZPointSet z(16);
    BBox3 region(Point3(0, 0, 0), Point3(16, 16, 16));
    std::vector<Point3> points;
    RNGSampler rng(new HaltonSequence(3), new HaltonSequence(5), new HaltonSequence(7));
    for (int i = 0; i < 1000; i++) {
      points.emplace_back(rng.sample(region));
      z.add(points[points.size() - 1]);
    }
    region.expand(4);
    z.buildAccelerationStructure();
    for (int i = 0; i < 1000; i++) {
      // generate a bounding box
      BBox3 r(rng.sample(region), rng.sample(region));
      std::vector<uint> result;
      z.search(r, [&](uint id) { result.emplace_back(id); });
      std::sort(result.begin(), result.end());
      std::vector<uint> bfResult;
      for (uint k = 0; k < points.size(); k++)
        if (r.contains(points[k]))
          bfResult.emplace_back(k);
      std::sort(bfResult.begin(), bfResult.end());
      EXPECT_EQ(result.size(), bfResult.size());
      for (uint k = 0; k < result.size(); k++)
        EXPECT_EQ(result[k], bfResult[k]);
    }
  }
  {
    ZPointSet z(16);
    BBox3 region(Point3(0, 0, 0), Point3(16, 16, 16));
    std::vector<Point3> points;
    RNGSampler rng(new HaltonSequence(3), new HaltonSequence(5), new HaltonSequence(7));
    for (int i = 0; i < 16; i++)
      for (int j = 0; j < 16; j++)
        for (int k = 0; k < 16; k++) {
          points.emplace_back(i, j, k);
          z.add(points[points.size() - 1]);
        }
    region.expand(4);
    for (int i = 0; i < 1000; i++) {
      // generate a bounding box
      BBox3 r(rng.sample(region), rng.sample(region));
      std::vector<uint> result;
      z.search(r, [&](uint id) { result.emplace_back(id); });
      std::sort(result.begin(), result.end());
      std::vector<uint> bfResult;
      for (uint k = 0; k < points.size(); k++)
        if (r.contains(points[k]))
          bfResult.emplace_back(k);
      std::sort(bfResult.begin(), bfResult.end());
      EXPECT_EQ(result.size(), bfResult.size());
      for (uint k = 0; k < result.size(); k++)
        EXPECT_EQ(result[k], bfResult[k]);
    }
  }
}

TEST(ZPointSet, Remove) {
  {
    BBox3 region(Point3(0, 0, 0), Point3(16, 16, 16));
    ZPointSet z(16);
    RNGSampler rng(new HaltonSequence(3), new HaltonSequence(5), new HaltonSequence(7));
    std::vector<Point3> points;
    for (uint j = 0; j < 400; j++) {
      auto p = rng.sample(region);
      z.add(p);
      points.emplace_back(p);
    }
    EXPECT_EQ(z.size(), 400u);
    for (uint i = 0; i < 400; i++)
      z.remove(i);
    EXPECT_EQ(z.size(), 0u);
    for (uint i = 0; i < 200; i++)
      z.add(rng.sample(region));
    EXPECT_EQ(z.size(), 200u);
  }
  {
    ZPointSet z(16);
    BBox3 region(Point3(0, 0, 0), Point3(16, 16, 16));
    RNGSampler rng(new HaltonSequence(3), new HaltonSequence(5), new HaltonSequence(7));
    for (uint j = 0; j < 200; j++)
      z.add(rng.sample(region));
    z.update();
    std::vector<uint> indices;
    z.iteratePoints([&](uint id, Point3 p) {
      UNUSED_VARIABLE(p);
      indices.emplace_back(id);
    });
    EXPECT_EQ(indices.size(), 200u);
    for (uint i = 0; i < 100; i++)
      z.remove(i);
    z.update();
    uint j = 0;
    z.iteratePoints([&](uint i, Point3 p) {
      static unsigned int k = 0;
      UNUSED_VARIABLE(p);
      while (indices[k] < 100)
        k++;
      EXPECT_EQ(indices[k], i);
      k++;
      j++;
    });
    EXPECT_EQ(j, 100u);
    EXPECT_EQ(z.size(), 100u);
  }
}

TEST(ZPointSet, SetPosition) {
  {
    ZPointSet z(16);
    BBox3 region(Point3(0, 0, 0), Point3(16, 16, 16));
    std::vector<Point3> points;
    RNGSampler rng(new HaltonSequence(3), new HaltonSequence(5), new HaltonSequence(7));
    for (int i = 0; i < 1000; i++) {
      points.emplace_back(rng.sample(region));
      z.add(points[points.size() - 1]);
    }
    region.expand(4);
    for (int i = 0; i < 1000; i++) {
      // generate a bounding box
      BBox3 r(rng.sample(region), rng.sample(region));
      std::vector<uint> result;
      z.search(r, [&](uint id) { result.emplace_back(id); });
      std::sort(result.begin(), result.end());
      std::vector<uint> bfResult;
      for (uint k = 0; k < points.size(); k++)
        if (r.contains(points[k]))
          bfResult.emplace_back(k);
      std::sort(bfResult.begin(), bfResult.end());
      EXPECT_EQ(result.size(), bfResult.size());
      for (uint k = 0; k < result.size(); k++)
        EXPECT_EQ(result[k], bfResult[k]);
    }
    // now change positions
    region.expand(-4);
    for (uint i = 0; i < 1000; i++) {
      points[i] = rng.sample(region);
      z.setPosition(i, points[i]);
    }
    region.expand(4);
    for (int i = 0; i < 1000; i++) {
      // generate a bounding box
      BBox3 r(rng.sample(region), rng.sample(region));
      std::vector<uint> result;
      z.search(r, [&](uint id) { result.emplace_back(id); });
      std::sort(result.begin(), result.end());
      std::vector<uint> bfResult;
      for (uint k = 0; k < points.size(); k++)
        if (r.contains(points[k]))
          bfResult.emplace_back(k);
      std::sort(bfResult.begin(), bfResult.end());
      EXPECT_EQ(result.size(), bfResult.size());
      for (uint k = 0; k < result.size(); k++)
        EXPECT_EQ(result[k], bfResult[k]);
    }
  }
}

// TODO: extreme cases