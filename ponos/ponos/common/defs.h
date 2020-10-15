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
/// \file defs.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date 2019-08-16
///
/// \brief type definitions

#ifndef PONOS_COMMON_DEFS_H
#define PONOS_COMMON_DEFS_H

#include <cstdint>
#include <type_traits>

#ifdef PONOS_DOUBLE
using real_t = double;
#else
using real_t = float;
#endif

using f32 = float;
using f64 = double;

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using ulong = unsigned long;
using uint = unsigned int;
using ushort = unsigned short;
using uchar = unsigned char;

using byte = uint8_t;

namespace ponos {

enum class DataType {
  I8,
  I16,
  I32,
  I64,
  U8,
  U16,
  U32,
  U64,
  F16,
  F32,
  F64,
  CUSTOM
};

template<typename T>
DataType dataTypeFrom() {
#define MATCH_TYPE(Type, R) \
  if(std::is_same_v<T, Type>) \
    return DataType::R;
  MATCH_TYPE(i8, I8)
  MATCH_TYPE(i16, I16)
  MATCH_TYPE(i32, I32)
  MATCH_TYPE(i64, I64)
  MATCH_TYPE(u8, U8)
  MATCH_TYPE(u16, U16)
  MATCH_TYPE(u32, U32)
  MATCH_TYPE(u64, U64)
  MATCH_TYPE(f32, F32)
  MATCH_TYPE(f64, F64)
  return DataType::CUSTOM;
#undef MATCH_TYPE
}

}

#endif
