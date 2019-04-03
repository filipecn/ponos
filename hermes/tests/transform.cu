#include <gtest/gtest.h>
#include <hermes/hermes.h>

using namespace hermes::cuda;

TEST(Transform, operators) {
  {
    auto t = scale(2.f, 10.f, -3.f);
    vec3f v(1.f);
    EXPECT_EQ(t(v), vec3f(2.f, 10.f, -3.f));
  }
  {
    vec3f trans(-14.f, 2.f, 0.f);
    auto t = translate(trans);
    point3f a(1.f, -1.f, 10.f);
    EXPECT_EQ(t(a), a + trans);
  }
  {
    vec3f offset(-0.5f, 0.f, 0.f);
    float dx = 1.f / 10.f;
    auto t = scale(dx, dx, dx) * translate(offset);
    EXPECT_EQ(t(point3f()), point3f(-0.5f * dx, 0.f, 0.f));
    EXPECT_EQ(t(point3f(10.f)), point3f(-0.5f * dx + 1, 1.f, 1.f));
  }
}

TEST(Transform, inverse) {
  {
    auto t = scale(2.f, 10.f, -3.f);
    auto inv = inverse(t);
    vec3f v(1.f);
    EXPECT_EQ(v, inv(t(v)));
  }
  {
    vec3f trans(-14.f, 2.f, 0.f);
    auto t = translate(trans);
    auto inv = inverse(t);
    point3f a(1.f, -1.f, 10.f);
    EXPECT_EQ(a, inv(t(a)));
  }
  {
    vec3f offset(-0.5f, 0.f, 0.f);
    float dx = 1.f / 10.f;
    auto t = scale(dx, dx, dx) * translate(offset);
    auto tinv = inverse(t);
    EXPECT_EQ(point3f(), tinv(t(point3f())));
    EXPECT_EQ(point3f(10.f), tinv(t(point3f(10.f))));
  }
}