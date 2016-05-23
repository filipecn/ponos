#include <aergia.h>
#include <poseidon.h>

int main() {
  poseidon::ParticleGrid pg;
  aergia::GraphicsDisplay& gd = aergia::createGraphicsDisplay(800, 800, "Hello Aergia");
  gd.start();
  return 0;
}
