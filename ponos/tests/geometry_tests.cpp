#include <catch2/catch.hpp>

#include <ponos/ponos.h>

using namespace ponos;

TEST_CASE("Point", "[geometry][point]") {
  SECTION("Point2") {
    SECTION("hash") {
      point2 a(0.00001, 0.1);
      point2 b(0.00001, 0.1);
      point2 c(0.0000000001, 0.1);
      REQUIRE(std::hash<point2>()(a) == std::hash<point2>()(b));
      REQUIRE(std::hash<point2>()(a) != std::hash<point2>()(c));
    }
  }SECTION("Point3") {
    SECTION("hash") {
      point3 a(0.00001, 0.1, 0.000000001);
      point3 b(0.00001, 0.1, 0.000000001);
      point3 c(0.0000000001, 0.1, 0.000000001);
      REQUIRE(std::hash<point3>()(a) == std::hash<point3>()(b));
      REQUIRE(std::hash<point3>()(a) != std::hash<point3>()(c));
    }
  }
}

TEST_CASE("Vector", "[geometry][vector]") {
  SECTION("Vector2") {
    SECTION("hash") {
      vec2 a(0.00001, 0.1);
      vec2 b(0.00001, 0.1);
      vec2 c(0.0000000001, 0.1);
      REQUIRE(std::hash<vec2>()(a) == std::hash<vec2>()(b));
      REQUIRE(std::hash<vec2>()(a) != std::hash<vec2>()(c));
    }
  }SECTION("Vector3") {
    SECTION("hash") {
      vec3 a(0.00001, 0.1, 0.000000001);
      vec3 b(0.00001, 0.1, 0.000000001);
      vec3 c(0.0000000001, 0.1, 0.000000001);
      REQUIRE(std::hash<vec3>()(a) == std::hash<vec3>()(b));
      REQUIRE(std::hash<vec3>()(a) != std::hash<vec3>()(c));
    }
  }
}

TEST_CASE("BBox3", "[geometry][bbox]") {
  SECTION("union") {
    bbox3 a(point3(), point3(1));
    bbox3 b(point3(-1), point3());
    bbox3 c = make_union(a, b);
    point3 l(-1), u(1);
    REQUIRE(c.lower == l);
    REQUIRE(c.upper == u);
  }SECTION("access") {
    auto l = point3(0, 1, 2);
    auto u = point3(3, 4, 3);
    bbox3 b(l, u);
    REQUIRE(b.lower == l);
    REQUIRE(b.upper == u);
    REQUIRE(b[0] == l);
    REQUIRE(b[1] == u);
    REQUIRE(b.corner(0) == point3(0, 1, 2));
    REQUIRE(b.corner(1) == point3(3, 1, 2));
    REQUIRE(b.corner(2) == point3(0, 4, 2));
    REQUIRE(b.corner(3) == point3(3, 4, 2));
    REQUIRE(b.corner(4) == point3(0, 1, 3));
    REQUIRE(b.corner(5) == point3(3, 1, 3));
    REQUIRE(b.corner(6) == point3(0, 4, 3));
    REQUIRE(b.corner(7) == point3(3, 4, 3));
  }
}

TEST_CASE("Matrix", "[geometry]") {
  SECTION("Identity") {
    mat4 m(true);
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        if (r == c)
          REQUIRE(m[r][c] == Approx(1).epsilon(1e-8));
        else
          REQUIRE(m[r][c] == Approx(0).epsilon(1e-8));
    REQUIRE(m.isIdentity());
  }
}

TEST_CASE("Transform", "[geometry][transform]") {
  SECTION("Sanity") {
    Transform t;
    REQUIRE(t.matrix().isIdentity());
  }//
  SECTION("orthographic projection") {
    SECTION("left handed") {
      auto t = Transform::ortho(-1, 1, -1, 1, -1, 1);
      REQUIRE(t(vec3(1, 0, 0)) == vec3(1, 0, 0));
      REQUIRE(t(vec3(0, 1, 0)) == vec3(0, 1, 0));
      REQUIRE(t(vec3(0, 0, 1)) == vec3(0, 0, 1));
    }//
    SECTION("right handed") {
      auto t = Transform::ortho(-1, 1, -1, 1, -1, 1,
                                transform_options::right_handed);
      REQUIRE(t(vec3(1, 0, 0)) == vec3(1, 0, 0));
      REQUIRE(t(vec3(0, 1, 0)) == vec3(0, 1, 0));
      REQUIRE(t(vec3(0, 0, 1)) == vec3(0, 0, -1));
    }//
    SECTION("cube") {
      auto t = Transform::ortho(-10, 10, -20, 20, -1, 1);
      REQUIRE(t.matrix() == mat4(
          0.1, 0, 0, 0,//
          0, 0.05, 0, 0,//
          0, 0, 1, 0,//
          0, 0, 0, 1//
      ));
    }//
    SECTION("zero to one") {
      auto t = Transform::ortho(-10, 10, -20, 20, -1, 1,
                                transform_options::right_handed |
                                    transform_options::zero_to_one);
      REQUIRE(t.matrix() == mat4(
          0.1, 0, 0, -0,//
          0, 0.05, 0, -0,//
          0, 0, -0.5, 0.5,//
          0, 0, 0, 1//
      ));
    }//
  }//
  SECTION("perspective projection") {
    SECTION("LH"){
      auto t = Transform::perspective(90, 1, 1, 10);
      auto y_scale= 1.f / std::tan(Trigonometry::degrees2radians(90) * 0.5);
//      REQUIRE(t.matrix() == mat4(
//          y_scale, 0, 0, 0,//
//          0, y_scale, 0, 0,//
//          0, 0, 11./9, -20./9,//
//          0, 0, 1, 0//
//      ));
    }//
    SECTION("RH"){
      auto t = Transform::perspective(90, 1, 1, 10, transform_options::right_handed);
      auto y_scale= 1.f / std::tan(Trigonometry::degrees2radians(90) * 0.5);
//      REQUIRE(t.matrix() == mat4(
//          y_scale, 0, 0, 0,//
//          0, y_scale, 0, 0,//
//          0, 0, -11./9, 20./9,//
//          0, 0, -1, 0//
//      ));
    }//
  }//
  SECTION("look at") {
    SECTION("left handed") {
      auto t = Transform::lookAt({1.f, 0.f, 0.f});
      REQUIRE(t.matrix() == mat4(
          0, 0, 1, 0,//
          0, 1, 0, 0,//
          -1, 0, 0, 1,//
          0, 0, 0, 1//
      ));
    }//
    SECTION("right handed") {
      auto t = Transform::lookAt({1.f, 0.f, 0.f}, {0, 0, 0}, {0, 1, 0}, transform_options::right_handed);
      REQUIRE(t.matrix() == mat4(
          0, 0, -1, 0,//
          0, 1, 0, 0,//
          1, 0, 0, 1,//
          0, 0, 0, 1//
      ));
    }//
  }//
}

TEST_CASE("Geometric Predicates", "[geometry]") {
  SECTION("ray_triangle") {
    {
      point3 a{6.46322, 0.400573, 5.2238};
      point3 b{6.85173, -0.39896, -1.24055};
      point3 c{8.03027, 3.73176, -0.479493};
      auto i = GeometricPredicates::intersect(a, b, c,
                                              {{20., 1., 0.},
                                               {-1., 0., 0.}});
      REQUIRE(i.has_value());
    }
    {
      point3 a = {-8.04074, 2.8569, -0.486101};
      point3 b = {-6.66026, -0.584077, -1.13166};
      point3 c = {-6.49687, 0.308156, 4.96059};
      auto i = GeometricPredicates::intersect(a, b, c,
                                              {{20., 1., 0.},
                                               {-1., 0., 0.}});
      REQUIRE(i.has_value());
    }
    real_t b0{}, b1{};
    point3 p0 = {-1., -1., 0.};
    point3 p1 = {1., -1., -0.};
    point3 p2 = {0., 1., 0.};
    auto i = GeometricPredicates::intersect(p0, p1, p2,
                                            {{0., 0., 1.},
                                             {0., 0., -1.}}, &b0, &b1);
    REQUIRE(i.has_value());
    REQUIRE(i.value() == Approx(1));
    REQUIRE(p0 * b0 + vec3(p1) * b1 + vec3(p2) * (1 - b0 - b1) == point3(0, 0, 0));
    i = GeometricPredicates::intersect(p0, p1, p2,
                                       {{-1., -1., 1.},
                                        {0., 0., -1.}}, &b0, &b1);
    REQUIRE(i.has_value());
    REQUIRE(i.value() == Approx(1));
    REQUIRE(p0 * b0 + vec3(p1) * b1 + vec3(p2) * (1 - b0 - b1) == point3(-1, -1, 0));
    i = GeometricPredicates::intersect(p0, p1, p2,
                                       {{0., -1., -1.},
                                        {0., 0., 1.}}, &b0, &b1);
    REQUIRE(i.has_value());
    REQUIRE(i.value() == Approx(1));
    REQUIRE(p0 * b0 + vec3(p1) * b1 + vec3(p2) * (1 - b0 - b1) == point3(0, -1, 0));
  }
}