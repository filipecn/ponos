#include <ponos.h>
#include <iostream>

int main() {
  ponos::ray r(ponos::point(0,0,0), ponos::vec3(1,2,3));
  ponos::point p = r(1.7);
  return 0;
}
