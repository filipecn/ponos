#include <poseidon/poseidon.h>
#include <gtest/gtest.h>

using namespace poseidon;

TEST(ParticleSystem, Iterate) {
  {
    ParticleSystem ps;
    for (int i = 0; i < 8; i++)
      ps.add(ponos::Point3(0, 0, 0));
    auto p1 = ps.addProperty<double>(1.);
    auto p2 = ps.addProperty<double>(2.);
    uint k = 0;
    ps.iterateParticles([&](ParticleSystem::ParticleAccessor p) {
      EXPECT_EQ(p.id(), k++);
      EXPECT_EQ(p.position(), ponos::Point3());
      p.property<double>(p1) = static_cast<double>(k);
      p.property<double>(p2) = static_cast<double>(k * 2);
    });
    k = 0;
    ps.iterateParticles([&](ParticleSystem::ParticleAccessor p) {
      k++;
      EXPECT_EQ(p.property<double>(p1), static_cast<double>(k));
      EXPECT_EQ(p.property<double>(p2), static_cast<double>(k * 2));
    });
  }
}

TEST(ParticleSystem, MaxMin) {
  ParticleSystem ps;
  auto p1 = ps.addProperty<double>(0.);
  auto p2 = ps.addProperty<double>(0.);
  for (size_t i = 0; i < 800; i++) {
    ps.add(ponos::Point3(0, 0, 0));
    ps.property<double>(i, p1) = i;
    ps.property<double>(i, p2) = -1.0 * i;
  }
  EXPECT_DOUBLE_EQ(ps.maxValue<double>(p1), 799.0);
  EXPECT_DOUBLE_EQ(ps.minValue<double>(p1), 0.0);
  EXPECT_DOUBLE_EQ(ps.maxValue<double>(p2), 0.0);
  EXPECT_DOUBLE_EQ(ps.minValue<double>(p2), -799.0);
  // TODO test other types of properties
}

TEST(ParticleSystem, Position) {
  ParticleSystem ps;
  for (int i = 0; i < 10; i++)
    ps.add(ponos::Point3(i, i, i));
  for (uint i = 0; i < 10; i++)
    EXPECT_EQ(ps.position(i), ponos::Point3(i, i, i));
  for (uint i = 0; i < 10; i++)
    ps.setPosition(i, ponos::Point3(i * 10, i, i));
  for (uint i = 0; i < 10; i++)
    EXPECT_EQ(ps.position(i), ponos::Point3(i * 10, i, i));
}

TEST(ParticleSystem, Gather) {
  {
    ParticleSystem ps;
    ps.add(ponos::Point3(0, 0, 0));
    ps.add(ponos::Point3(1, 0, 0));
    ps.add(ponos::Point3(1, 1, 0));
    ps.add(ponos::Point3(0, 1, 0));
    ps.add(ponos::Point3(0, 0, 1));
    ps.add(ponos::Point3(1, 0, 1));
    ps.add(ponos::Point3(1, 1, 1));
    ps.add(ponos::Point3(0, 1, 1));
    uint p1 = ps.addProperty<double>(1.);
    uint p2 = ps.addProperty<double>(2.);
    std::shared_ptr<ponos::RBFInterpolator<ponos::Point3, double>>
        interpolator(new ponos::RBFInterpolator<ponos::Point3, double>());
    EXPECT_DOUBLE_EQ(1., ps.gatherProperty<double>(ponos::Point3(0.5, 0.5, 0.5),
                                                   p1, 1., interpolator.get()));
    EXPECT_DOUBLE_EQ(2., ps.gatherProperty<double>(ponos::Point3(0.5, 0.5, 0.5),
                                                   p2, 1., interpolator.get()));
    auto v = ps.gatherProperties<double>(ponos::Point3(0.5,0.5,0.5), {p1, p2}, 1., interpolator.get());
    EXPECT_DOUBLE_EQ(1., v[0]);
    EXPECT_DOUBLE_EQ(2., v[1]);
  }
}