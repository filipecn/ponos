/// Copyright (c) 2019, FilipeCN.
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
///\file size.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2019-08-16
///
///\breaf Representation of sizes

#ifndef PONOS_COMMON_SIZE_H
#define PONOS_COMMON_SIZE_H

#include <ponos/common/defs.h>

namespace ponos {

/// Holds 2-dimensional size
///\tparam T must be an unsigned integer type
template <typename T> class Size2 {
  static_assert(std::is_same<T, u8>::value || std::is_same<T, u16>::value ||
                    std::is_same<T, u32>::value || std::is_same<T, u64>::value,
                "Size2 must hold an unsigned integer type!");

public:
  Size2(T width = T(0), T height = T(0)) : width(width), height(height) {}
  T operator[](int i) const { return (&width)[i]; }
  T &operator[](int i) { return (&width)[i]; }

  T width = T(0);
  T height = T(0);
};

/// Holds 2-dimensional size
///\tparam T must be an unsigned integer type
template <typename T> class Size3 {
  static_assert(std::is_same<T, u8>::value || std::is_same<T, u16>::value ||
                    std::is_same<T, u32>::value || std::is_same<T, u64>::value,
                "Size3 must hold an unsigned integer type!");

public:
  Size3(T width = T(0), T height = T(0), T depth = T(0))
      : width(width), height(height), depth(depth) {}
  T operator[](int i) const { return (&width)[i]; }
  T &operator[](int i) { return (&width)[i]; }

  T width = T(0);
  T height = T(0);
  T depth = T(0);
};

using size2 = Size2<u32>;
using size2_8 = Size2<u8>;
using size2_16 = Size2<u16>;
using size2_32 = Size2<u32>;
using size2_64 = Size2<u64>;
using size3 = Size3<u32>;
using size3_8 = Size3<u8>;
using size3_16 = Size3<u16>;
using size3_32 = Size3<u32>;
using size3_64 = Size3<u64>;

} // namespace ponos

#endif