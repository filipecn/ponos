/// Copyright (c) 2020, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
///\file file_system.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-06-09
///
///\brief

#include <ponos/random/noise.h>
#include <ponos/common/index.h>

namespace ponos {

#define TABSIZE 256
#define TABMASK (TABSIZE-1)
#define PERM(x) permutation_table[(x)STABMASK]
#define INDEX(ix, iy, iz) PERM((ix)+PERM((iy)+PERM(iz)))

static u8 permutation_table[TABSIZE] = {
    225, 155, 210, 108, 175, 199, 221, 144, 203, 116, 70, 213, 69, 158, 33, 252,
    5, 82, 173, 133, 222, 139, 174, 27, 9, 71, 90, 246, 75, 130, 91, 191,
    169, 138, 2, 151, 194, 235, 81, 7, 25, 113, 228, 159, 205, 253, 134, 142,
    248, 65, 224, 217, 22, 121, 229, 63, 89, 103, 96, 104, 156, 17, 201, 129,
    36, 8, 165, 110, 237, 117, 231, 56, 132, 211, 152, 20, 181, 111, 239, 218,
    170, 163, 51, 172, 157, 47, 80, 212, 176, 250, 87, 49, 99, 242, 136, 189,
    162, 115, 44, 43, 124, 94, 150, 16, 141, 247, 32, 10, 198, 223, 255, 72,
    53, 131, 84, 57, 220, 197, 58, 50, 208, 11, 241, 28, 3, 192, 62, 202,
    18, 215, 153, 24, 76, 41, 15, 179, 39, 46, 55, 6, 128, 167, 23, 188,
    106, 34, 187, 140, 164, 73, 112, 182, 244, 195, 227, 13, 35, 77, 196, 185,
    26, 200, 226, 119, 31, 123, 168, 125, 249, 68, 183, 230, 177, 135, 160, 180,
    12, 1, 243, 148, 102, 166, 38, 238, 251, 37, 240, 126, 64, 74, 161, 40,
    184, 149, 171, 178, 101, 66, 29, 59, 146, 61, 254, 107, 42, 86, 154, 4,
    236, 232, 120, 21, 233, 209, 45, 98, 193, 114, 78, 19, 206, 14, 118, 127,
    48, 79, 147, 85, 30, 207, 219, 54, 88, 234, 190, 122, 95, 67, 143, 109,
    137, 214, 145, 93, 92, 100, 245, 0, 216, 186, 60, 83, 105, 97, 204, 52
};

static u8 P[2 * TABSIZE];

static vec2 G2[4] = {{1, 1}, {-1, 1}, {1, -1}, {-1, -1}};

real_t Perlin::at(const point2 &p) {
  // Initialize array of gradient index permutations in first call
  static bool init_flag = true;
  if (init_flag) {
    init();
    init_flag = false;
  }
  // Each cubical vertex has a wavelet with the following properties:
  // - It has a value of zero at its center.
  // - It has some randomly chosen gradient at its center.
  // - It smoothly drops off to zero a unit distance from its center.
  // compute which cubical cell we are in
  index2 ij(p.x, p.y);
  // Each wavelet is a product of
  // - a cubic weight that drops to zero at radius 1
  // - a linear function, which is zero at [i,j]
  auto drop = [](real_t t) -> real_t {
    return t * t * t * (t * (t * 6 - 15) + 10);
  };
  // relative position to the wavelet center -1 <= x,y <= 1:
  vec2 xy(p.x - ij.i, p.y - ij.j);
  // mod ij by 256
  ij.i &= TABSIZE - 1;
  ij.j &= TABSIZE - 1;
  // compute the dropoff uv about ij
  vec2 uv(drop(xy.x), drop(xy.y));
  // Each index [i,j] is folded into a single number by calling P[P[P[i] + j] + k]
  // that is the hash to retrieve the gradient for that index
  vec2 g00 = G2[P[P[ij.i] + ij.j] % 4];
  vec2 g10 = G2[P[P[ij.i + 1] + ij.j] % 4];
  vec2 g01 = G2[P[P[ij.i] + ij.j + 1] % 4];
  vec2 g11 = G2[P[P[ij.i + 1] + ij.j + 1] % 4];
  // The value of the wavelet is given by the weight of the relative position
  // multiplied by the dot product of the gradient and the relative position
  real_t w00 = dot(g00, xy - vec2(0, 0));
  real_t w10 = dot(g10, xy - vec2(1, 0));
  real_t w01 = dot(g01, xy - vec2(0, 1));
  real_t w11 = dot(g11, xy - vec2(1, 1));
// bilinear interpolate wavelet values using weights
#define lerp_(t, a, b) (a + t * (b - a))
  return lerp_(uv.y, lerp_(uv.x, w00, w10), lerp_(uv.x, w01, w11));
}

void Perlin::init() {
  // the permutation array P is the permutation table repeated to avoid
  // mod operations in fold method
  for (int i = 0; i < TABSIZE; ++i)
    P[i] = P[i + TABSIZE] = permutation_table[i];
}

}