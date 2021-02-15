#include <catch2/catch.hpp>

#include <ponos/ponos.h>

using namespace ponos;
struct Primitive {
  [[nodiscard]] const bbox3 &bounds() const { return b; };
  bbox3 b;
  u32 id{};
};

TEST_CASE("BVH", "[structures]") {
  SECTION("Empty") {
    std::vector<Primitive> primitives;
    BVH<Primitive> bvh(primitives);
  }//
  SECTION("Single Primitive") {
    std::vector<Primitive> primitives = {{bbox3::unitBox(), 0u}};
    BVH<Primitive> bvh(primitives);
    // traverse and count leaves
    u64 leaf_count = 0;
    bvh.traverse([](const bbox3 &bounds) -> bool { return true; }, [&](const BVH<Primitive>::LinearNode &node) {
      leaf_count++;
    });
    REQUIRE(leaf_count == 1);
  }//
  SECTION("Grid") {
    std::vector<Primitive> primitives;
    u32 n = 2;
    for (u32 x = 0; x < n; ++x)
      for (u32 y = 0; y < n; ++y)
        for (u32 z = 0; z < n; ++z)
          primitives.push_back({{{x - .1f, y - .1f, z - .1f},
                                 {x + .1f, y + .1f, z + .1f}},
                                x * n * n + y * n + z});
    BVH<Primitive> bvh(primitives, 4);
    // traverse and count leaves
    u64 leaf_count = 0;
    bvh.traverse([](const bbox3 &bounds) -> bool { return true; }, [&](const BVH<Primitive>::LinearNode &node) {
      leaf_count++;
    });
    REQUIRE(leaf_count == n * n * n);
    ray3 r{{0.f, -1.f, 0.f}, {0.f, 1.f, 0.f}};
    auto primitive = bvh.intersect(r, [&](const Primitive &p) -> auto {
      return GeometricPredicates::intersect(p.bounds(), r);
    });
    REQUIRE(primitive.has_value());
    REQUIRE(primitive.value().id == 0u);
  }//
}