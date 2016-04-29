#include <ponos.h>
#include <iostream>

void member_access() {
  ponos::point v(1,2,3);
  for (int i = 0; i < 3; i++)
    CHECK_FLOAT_EQUAL(v[i], static_cast<float>(i+1));
}

int main() {
  member_access();
  return 0;
}
