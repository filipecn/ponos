#ifndef AERGIA_COLORS_COLOR_H
#define AERGIA_COLORS_COLOR_H

#include <ponos.h>

namespace aergia {

class Color {
public:
  Color() {
    r = g = b = 0.f;
    a = 1.f;
  }
  Color(const ponos::vec3 &v) : r(v.x), g(v.y), b(v.z), a(1.f) {}
  Color(float _r, float _g, float _b, float _a = 1.f)
      : r(_r), g(_g), b(_b), a(_a) {}
  const float *asArray() const { return &r; }
  float r, g, b, a;
};

inline Color mix(float t, const Color &a, const Color &b) {
  return Color(ponos::lerp(t, a.r, b.r), ponos::lerp(t, a.g, b.g),
               ponos::lerp(t, a.b, b.b));
}

#define COLOR_TRANSPARENT Color(0.f, 0.f, 0.f, 0.f);
#define COLOR_BLACK Color(0.f, 0.f, 0.f, 1.f)
#define COLOR_RED Color(1.f, 0.f, 0.f, 1.f)
#define COLOR_GREEN Color(0.f, 1.f, 0.f, 1.f)
#define COLOR_BLUE Color(0.f, 0.f, 1.f, 1.f)

} // aergia namespace

#endif // AERGIA_COLORS_COLOR_H
