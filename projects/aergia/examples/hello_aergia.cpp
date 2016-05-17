#include <aergia.h>

int main() {
  aergia::GraphicsDisplay& gd = aergia::createGraphicsDisplay(800, 800, "Hello Aergia");
  gd.start();
  return 0;
}
