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
///\file str.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-10-07
///
///\brief String utils

#include "str.h"

#include <iostream>
#include <utility>

namespace ponos {

std::string Str::join(const std::vector<std::string> &s, const std::string &separator) {
  std::string r;
  bool first = true;
  for (const auto &ss : s) {
    if (!first)
      r += separator;
    first = false;
    r += ss;
  }
  return r;
}

std::vector<std::string> Str::split(const std::string &s, const std::string &delimiters) {
  std::vector<std::string> tokens;

  if (s.empty())
    return tokens;

  std::string::size_type lastPos = s.find_first_not_of(delimiters, 0);
  std::string::size_type pos = s.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos) {
    tokens.push_back(s.substr(lastPos, pos - lastPos));
    lastPos = s.find_first_not_of(delimiters, pos);
    pos = s.find_first_of(delimiters, lastPos);
  }
  return tokens;
}

bool Str::match_r(const std::string &s, const std::string &pattern, std::regex_constants::match_flag_type flags) {
  std::smatch m;
  return std::regex_match(s, m, std::regex(pattern), flags);
}

bool Str::contains_r(const std::string &s, const std::string &pattern, std::regex_constants::match_flag_type flags) {
  std::smatch m;
  return std::regex_search(s, m, std::regex(pattern), flags);
}

std::smatch Str::search_r(const std::string &s,
                          const std::string &pattern,
                          std::regex_constants::match_flag_type flags) {
  std::smatch result;
  std::regex_search(s, result, std::regex(pattern), flags);
  return result;
}

bool Str::search_r(std::string s,
                   const std::string &pattern,
                   const std::function<void(const std::smatch &)> &callback,
                   std::regex_constants::match_flag_type flags) {
  std::smatch result;
  std::regex r(pattern);
  bool found = false;
  while (std::regex_search(s, result, r, flags)) {
    callback(result);
    s = result.suffix().str();
    found = true;
  }
  return found;
}

std::string Str::replace_r(const std::string &s,
                           const std::string &pattern,
                           const std::string &format,
                           std::regex_constants::match_flag_type flags) {
  return std::regex_replace(s, std::regex(pattern), format, flags);
}

Str::Str() = default;

Str::Str(std::string s) : s_{std::move(s)} {}

Str::Str(const char *s) : s_{s} {}

Str::Str(Str &&other) noexcept: s_{std::move(other.s_)} {}

Str::Str(const Str &other) = default;

Str::~Str() = default;

}

