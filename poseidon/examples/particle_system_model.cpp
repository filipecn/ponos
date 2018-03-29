// Created by filipecn on 2/26/18.
#include <aergia/aergia.h>
#include <poseidon/poseidon.h>

int main() {
  aergia::SceneApp<> app(800, 800);
  app.init();
  ponos::BBox region = ponos::scale(16, 16, 16)(ponos::BBox::unitBox());
  poseidon::ParticleSystem ps;
  auto p1 = ps.addProperty<double>(0.);
  ponos::RNGSampler rng;
  for (size_t i = 0; i < 2000; i++)
    ps.add(rng.sample(region));
  ps.iterateParticles([p1](poseidon::ParticleSystem::ParticleAccessor acc) {
    acc.property<double>(p1) = acc.position().x;
  });
  poseidon::ParticleSystemModel psm(ps, 0.1);
  psm.addPropertyColor<double>(p1,aergia::HEAT_GREEN_COLOR_PALETTE);
  psm.selectProperty<double>(p1);
  psm.update();
  app.scene.add(&psm);
  aergia::SceneObjectSPtr grid(new aergia::CartesianGrid(5));
  app.scene.add(grid.get());
  app.run();
}
