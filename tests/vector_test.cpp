#include <ponos.h>
#include <iostream>

void member_access() {
  ponos::vec3 v(1,2,3);
  for (int i = 0; i < 3; i++)
    CHECK_FLOAT_EQUAL(v[i], static_cast<float>(i+1));
}

void arithmetic() {
  ponos::vec3 a;
  ponos::vec3 b(1,2,3);
  a += b;
  for (int i = 0; i < 3; i++)
    CHECK_FLOAT_EQUAL(a[i], static_cast<float>(i+1));
  ponos::vec2 c;
}

int main() {
  member_access();
  return 0;
}
