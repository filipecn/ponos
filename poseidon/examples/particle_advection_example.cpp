// Created by filipecn on 2/27/18.
#include <aergia/aergia.h>
#include <poseidon/poseidon.h>

int main() {
  // generate vector field represented by particles
  poseidon::ParticleSystem vectorField;
  vectorField.addProperty<double>(0.); // vx
  vectorField.addProperty<double>(0.); // vy
  vectorField.addProperty<double>(0.); // vz
  vectorField.iterateParticles([](poseidon::ParticleSystem::ParticleAccessor acc) {
    auto v = poseidon::enrightField(acc.position());
    for (size_t i = 0; i < 3; i++)
      acc.property<double>(i) = v[i];
  });
  // advection scheme
  ponos::RBFInterpolator<ponos::Point3, double> interpolator;
  std::function<ponos::Vector3(ponos::Point3)> velocity = [&](ponos::Point3 p) -> ponos::Vector3 {
    auto v0_ = poseidon::enrightField(p);
    return ponos::Vector3(v0_[0], v0_[1], v0_[2]);
    //auto v = vectorField.gatherProperties(p, {0,1,2}, 0.1, &interpolator);
    //return ponos::Vector3(v[0], v[1], v[2]);
  };
  std::function<ponos::Point3(double, ponos::Point3)> advect = [&](double dt,
                                                                   ponos::Point3
                                                                   p) {
    ponos::Point3 np = p;
    double step = dt / 2.;
    for (int t = 0; t < 2; t++) {
      // Mid-point rule
      auto v0 = velocity(p);
      ponos::Point3 midPt = np + 0.5 * step * v0;
      auto midVel = velocity(midPt);
      np += step * midVel;
    }
    return np;
  };
  // set particles
  ponos::BBox region = ponos::BBox::unitBox();
  poseidon::ParticleSystem ps;
  auto p1 = ps.addProperty<double>(0.);
  ponos::RNGSampler rng;
  for (size_t i = 0; i < 2000; i++)
    ps.add(rng.sample(region));
  aergia::SceneApp<> app(800, 800, "", false);
  app.init();
  app.addViewport(0,0,800,800);
  app.getCamera<aergia::UserCamera>(0)->setPosition(ponos::Point3(2));
  poseidon::ParticleSystemModel psm(ps, 0.005);
  psm.addPropertyColor<double>(p1, aergia::HEAT_GREEN_COLOR_PALETTE);
  psm.selectProperty<double>(p1);
  app.scene.add(&psm);
  aergia::SceneObjectSPtr grid(new aergia::CartesianGrid(5));
  app.scene.add(grid.get());
  app.renderCallback = [&]() {
    ps.iterateParticles([&](poseidon::ParticleSystem::ParticleAccessor acc) {
      ps.setPosition(acc.id(), advect(0.001, ps.position(acc.id())));
      acc.property<double>(p1) = velocity(acc.position()).length2();
    });
    psm.update();
  };
  app.run();
}

