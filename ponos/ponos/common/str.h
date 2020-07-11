//// Copyright (c) 2020, FilipeCN.
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
///\file str.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-10-07
///
///\brief String utils

#ifndef PONOS_PONOS_PONOS_COMMON_STR_H
#define PONOS_PONOS_PONOS_COMMON_STR_H

#include <string>
#include <sstream>
#include <vector>

namespace ponos {

class Str {
  // Condition to stop processing
  static void concat(std::ostringstream &s) { (void(s)); }

  template <typename H, typename... T>
  static void concat(std::ostringstream &s, H p, T... t) {
    s << p;
    concat(s, t...);
  }
public:
  /// Concatenates multiple elements into a single string.
  /// \tparam Args
  /// \param args
  /// \return a single string of the resulting concatenation
  template <class... Args>
  static std::string concat(const Args &... args) {
    std::ostringstream s;
    concat(s, args...);
    return s.str();
  }
  /// Splits a string into tokens separated by delimiters
  /// \param s **[in]** input string
  /// \param delimiters **[in | default = " "]** delimiters
  /// \return a vector of substrings
  static std::vector<std::string> split(const std::string& s,
      const std::string& delimiters = " ");
};

}

#endif //PONOS_PONOS_PONOS_COMMON_STR_H
