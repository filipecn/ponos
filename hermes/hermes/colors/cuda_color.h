/*
 * Copyright (c) 2019 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * iM the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
#ifndef HERMES_COLORS_CUDA_COLOR_H
#define HERMES_COLORS_CUDA_COLOR_H

#include <hermes/geometry/cuda_numeric.h>
#include <hermes/geometry/vector.h>

namespace hermes {

namespace cuda {

class Color {
public:
  __host__ __device__ Color() {
    r = g = b = 0.f;
    a = 1.f;
  }
  __host__ __device__ Color(const vec3 &v) : r(v.x), g(v.y), b(v.z), a(1.f) {}
  __host__ __device__ Color(float _r, float _g, float _b, float _a = 1.f)
      : r(_r), g(_g), b(_b), a(_a) {}
  static __host__ __device__ Color transparent() {
    return {0.f, 0.f, 0.f, 0.f};
  }
  static __host__ __device__ Color black(float alpha = 1.f) {
    return {0.f, 0.f, 0.f, alpha};
  }
  static __host__ __device__ Color white(float alpha = 1.f) {
    return {1.f, 1.f, 1.f, alpha};
  }
  static __host__ __device__ Color red(float alpha = 1.f) {
    return {1.f, 0.f, 0.f, alpha};
  }
  static __host__ __device__ Color green(float alpha = 1.f) {
    return {0.f, 1.f, 0.f, alpha};
  }
  static __host__ __device__ Color blue(float alpha = 1.f) {
    return {0.f, 0.f, 1.f, alpha};
  }
  static __host__ __device__ Color purple(float alpha = 1.f) {
    return {1.f, 0.f, 1.f, alpha};
  }
  float r, g, b, a = 1.f;
};
/// \param rbg [0..255]
/// \return Color [0..1]
inline __host__ __device__ Color tonRGB(const Color &rgb) {
  return {rgb.r / 255, rgb.g / 255, rgb.b / 255, rgb.a / 255};
}
/// \param nRGB [0..1]
/// \return Color [0..255]
inline __host__ __device__ Color toRGB(const Color &nRGB) {
  int r = 255.9999f * nRGB.r;
  int g = 255.9999f * nRGB.g;
  int b = 255.9999f * nRGB.b;
  int a = 255.9999f * nRGB.a;
  return {static_cast<float>(r), static_cast<float>(g), static_cast<float>(b),
          static_cast<float>(a)};
}
/// \param nRGB linear color [0..1]
/// \return Color in sRGB space [0..1]
inline __host__ __device__ Color nRGB2sRGB(const Color &nRGB) {
  auto f = [](float u) -> float {
    return (u <= 0.0031308f) ? 12.92f * u : 1.005f * powf(u, 1 / 2.4f) - 0.055f;
  };
  return {f(nRGB.r), f(nRGB.g), f(nRGB.b), nRGB.a};
}
/// \param sRGB color in sRGB space [0..1]
/// \return Color in linear rgb space [0..1]
inline __host__ __device__ Color sRGB2nRGB(const Color &sRGB) {
  auto f = [](float u) -> float {
    return (u <= 0.04045f) ? u / 12.92f : powf((u + 0.055f) / 1.055f, 2.4f);
  };
  return {f(sRGB.r), f(sRGB.g), f(sRGB.b), sRGB.a};
}
/// \param a
/// \param b
/// \param t
/// \return Color
inline __host__ __device__ Color mix(const Color &a, const Color &b, float t) {
  return Color(lerp(t, a.r, b.r), lerp(t, a.g, b.g), lerp(t, a.b, b.b));
}
/// \param nRGB linear color [0..1]
/// \param gamma
/// \return float
inline __host__ __device__ float brightnessFromnRGB(const Color &nRGB,
                                                    float gamma) {
  return powf(nRGB.r + nRGB.g + nRGB.b, gamma);
}
/// \param a
/// \param b
/// \param t
/// \return Color
inline __host__ __device__ Color mixsRGB(const Color &a, const Color &b,
                                         float t) {
  // convert to nRGB
  auto sA = sRGB2nRGB(a);
  auto sB = sRGB2nRGB(b);
  // compute brightness and intensity
  float gamma = .43;
  auto brightA = brightnessFromnRGB(sA, gamma);
  auto brightB = brightnessFromnRGB(sB, gamma);
  auto brightness = lerp(t, brightA, brightB);
  float intensity = powf(brightness, 1 / gamma);
  // compute interpolated color
  Color c = mix(sA, sB, t);
  // adjust values
  if (c.r + c.g + c.b != 0) {
    float k = intensity / (c.r + c.g + c.b);
    c.r *= k;
    c.g *= k;
    c.b *= k;
  }
  return nRGB2sRGB(c);
}

} // namespace cuda

} // namespace hermes

#endif // CIRCE_COLORS_COLOR_H
