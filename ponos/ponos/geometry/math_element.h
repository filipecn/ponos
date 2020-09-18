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
///\file math_element.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-17-09
///
///\brief

#ifndef PONOS_PONOS_PONOS_GEOMETRY_MATH_ELEMENT_H
#define PONOS_PONOS_PONOS_GEOMETRY_MATH_ELEMENT_H

#include <ponos/common/defs.h>

namespace ponos {

/// Interface used by all basic geometric entities, such as point and vectors.
template<typename NUMERIC_TYPE, u64 COMPONENT_COUNT>
class MathElement {
public:
  NUMERIC_TYPE numeric_data;
  static inline constexpr u64 componentCount() { return COMPONENT_COUNT; };
  static inline constexpr u64 numericTypeSizeInBytes() { return sizeof(NUMERIC_TYPE); };
};

}

#endif //PONOS_PONOS_PONOS_GEOMETRY_MATH_ELEMENT_H
