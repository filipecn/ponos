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
#include <string>

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

enum class DataType : u8 {
  I8 = 0,
  I16 = 1,
  I32 = 2,
  I64 = 3,
  U8 = 4,
  U16 = 5,
  U32 = 6,
  U64 = 7,
  F16 = 8,
  F32 = 9,
  F64 = 10,
  CUSTOM = 11
};

class DataTypes {
public:
  static DataType typeFrom(u8 index) {
#define MATCH_TYPE(Type) \
  if((u8)DataType::Type == index) \
    return DataType::Type;
    MATCH_TYPE(I8)
    MATCH_TYPE(I16)
    MATCH_TYPE(I32)
    MATCH_TYPE(I64)
    MATCH_TYPE(U8)
    MATCH_TYPE(U16)
    MATCH_TYPE(U32)
    MATCH_TYPE(U64)
    MATCH_TYPE(F32)
    MATCH_TYPE(F64)
    return DataType::CUSTOM;
#undef MATCH_TYPE
  }
  template<typename T>
  static DataType typeFrom() {
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
  static std::string typeName(DataType type) {
#define DATA_TYPE_NAME(Type) \
      if(DataType::Type == type) \
    return #Type;
    DATA_TYPE_NAME(I8)
    DATA_TYPE_NAME(I16)
    DATA_TYPE_NAME(I32)
    DATA_TYPE_NAME(I64)
    DATA_TYPE_NAME(U8)
    DATA_TYPE_NAME(U16)
    DATA_TYPE_NAME(U32)
    DATA_TYPE_NAME(U64)
    DATA_TYPE_NAME(F16)
    DATA_TYPE_NAME(F32)
    DATA_TYPE_NAME(F64)
    DATA_TYPE_NAME(CUSTOM)
    return "CUSTOM";
#undef DATA_TYPE_NAME
  }
};

}

#endif
