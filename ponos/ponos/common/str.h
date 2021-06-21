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
#include <regex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <ponos/common/defs.h>

namespace ponos {

class Str {
  template<typename T>
  static void format_r_(std::stringstream &s, const std::string &fmt, u32 i, const T &first) {
    auto first_i = i;
    while (i + 1 < fmt.size() && !(fmt[i] == '{' && fmt[i + 1] == '}'))
      ++i;
    if (i + 1 < fmt.size()) {
      s << fmt.substr(first_i, i - first_i);
      s << first;
      s << fmt.substr(i + 2, fmt.size() - i - 2);
    } else
      s << fmt.substr(first_i, fmt.size() - first_i);
  }

  template<typename T, typename...Ts>
  static void format_r_(std::stringstream &s, const std::string &fmt, u32 i, const T &first, Ts &&... rest) {
    // iterate until first occurrence of pair {}
    auto first_i = i;
    while (i + 1 < fmt.size() && !(fmt[i] == '{' && fmt[i + 1] == '}'))
      ++i;
    if (i + 1 < fmt.size()) {
      s << fmt.substr(first_i, i - first_i);
      s << first;
      if constexpr(sizeof ...(rest) > 0)
        format_r_(s, fmt, i + 2, std::forward<Ts>(rest)...);
      else
        s << fmt.substr(i + 2, fmt.size() - i - 2);
    } else
      s << fmt.substr(first_i, fmt.size() - first_i);
  }

public:
  // ***********************************************************************
  //                           STATIC METHODS
  // ***********************************************************************
  ///
  /// \tparam Ts
  /// \param fmt
  /// \param args
  /// \return
  template<typename... Ts>
  static std::string format(const std::string &fmt, Ts &&... args) {
    std::stringstream s;
    std::string r;
    if constexpr(sizeof...(args) > 0) {
      format_r_(s, fmt, 0, std::forward<Ts>(args) ...);
      r = s.str();
    }
    else
      r = fmt;
    return r;
  }
  template<typename T>
  static std::string toHex(T i, bool leading_zeros = false, bool zero_x = false) {
    std::stringstream stream;
    if (zero_x)
      stream << "0x";
    if (leading_zeros)
      stream << std::setfill('0') << std::setw(sizeof(T) * 2)
             << std::hex << i;
    else
      stream << std::hex << i;
    if (!i)
      stream << '0';
    return stream.str();
  }
  /// Concatenates multiple elements_ into a single string.
  /// \tparam Args
  /// \param args
  /// \return a single string of the resulting concatenation
  template<class... Args>
  static std::string concat(const Args &... args) {
    std::stringstream s;
    (s << ... << args);
    return s.str();
  }
  /// Concatenate strings together separated by a separator
  /// \param s array of strings
  /// \param separator **[in | ""]**
  /// \return final string
  static std::string join(const std::vector<std::string> &v, const std::string &separator = "");
  template<typename T>
  static std::string join(const std::vector<T> &v, const std::string &separator = "") {
    bool first = true;
    std::stringstream r;
    for (const auto &s : v) {
      if (!first)
        r << separator;
      first = false;
      r << s;
    }
    return r.str();
  }
  /// Splits a string into tokens separated by delimiters
  /// \param s **[in]** input string
  /// \param delimiters **[in | default = " "]** delimiters
  /// \return a vector of substrings
  static std::vector<std::string> split(const std::string &s,
                                        const std::string &delimiters = " ");
  /// Checks if a string s matches exactly a regular expression
  /// \param s input string
  /// \param pattern regex pattern
  /// \param flags [optional] controls how pattern is matched
  /// \return true if s matches exactly the pattern
  static bool match_r(const std::string &s, const std::string &pattern,
                      std::regex_constants::match_flag_type flags = std::regex_constants::match_default);
  /// Checks if any substring of s matches a regular expression
  /// \param s input string
  /// \param pattern regex pattern
  /// \param flags [optional] controls how pattern is matched
  /// \return true if s contains the pattern
  static bool contains_r(const std::string &s, const std::string &pattern,
                         std::regex_constants::match_flag_type flags = std::regex_constants::match_default);
  /// Search the first substrings of s that matches the pattern
  /// \param s input string
  /// \param pattern regular expression pattern
  /// \param flags [optional] controls how pattern is matched
  /// \return std match object containing the first match
  static std::smatch search_r(const std::string &s, const std::string &pattern,
                              std::regex_constants::match_flag_type flags = std::regex_constants::match_default);
  /// Iterate over all substrings of s that match the pattern
  /// \param s input string
  /// \param pattern regular expression pattern
  /// \param callback called for each match
  /// \param flags [optional] controls how pattern is matched
  /// \return true if any match occurred
  static bool search_r(std::string s,
                       const std::string &pattern,
                       const std::function<void(const std::smatch &)> &callback,
                       std::regex_constants::match_flag_type flags = std::regex_constants::match_default);
  /// Replaces all matches of pattern in s by format
  /// \param s input string
  /// \param pattern regular expression pattern
  /// \param format replacement format
  /// \param flags [optional] controls how pattern is matched and how format is replaced
  /// \return A copy of s with all replacements
  static std::string replace_r(const std::string &s, const std::string &pattern, const std::string &format,
                               std::regex_constants::match_flag_type flags = std::regex_constants::match_default);
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  /// \param s
  Str();
  Str(std::string s);
  Str(const char *s);
  Str(const Str &other);
  Str(Str &&other) noexcept;
  ~Str();
  // ***********************************************************************
  //                             ACCESS
  // ***********************************************************************
  [[nodiscard]] inline const std::string &str() const { return s_; }
  [[nodiscard]] inline const char *c_str() const { return s_.c_str(); }
  // ***********************************************************************
  //                             METHODS
  // ***********************************************************************
  template<class... Args>
  void append(const Args &... args) {
    std::ostringstream s;
    (s << ... << args);
    s_ += s.str();
  }
  template<class... Args>
  void appendLine(const Args &... args) {
    std::ostringstream s;
    (s << ... << args);
    s_ += s.str() + '\n';
  }
  // ***********************************************************************
  //                             OPERATORS
  // ***********************************************************************
//  operator const std::string &() const { return s_; }
  Str &operator=(const Str &s) = default;
  template<typename T>
  Str &operator=(const T &t) {
    std::stringstream ss;
    ss << t;
    s_ = ss.str();
    return *this;
  }

  Str &operator+=(const Str &other) {
    s_ += other.s_;
    return *this;
  }

  template<typename T>
  Str &operator+=(const T &t) {
    std::stringstream ss;
    ss << t;
    s_ += ss.str();
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &os, const Str &s) {
    os << s.str();
    return os;
  }
private:
  std::string s_;
};

inline bool operator==(const Str &s, const char *ss) {
  return s.str() == ss;
}

inline bool operator==(const char *ss, const Str &s) {
  return s.str() == ss;
}

template<typename T>
inline bool operator==(const T &t, const Str &s) {
  std::stringstream ss;
  ss << t;
  return s.str() == ss.str();
}

template<typename T>
inline bool operator==(const Str &s, const T &t) {
  std::stringstream ss;
  ss << t;
  return s.str() == ss.str();
}

template<typename T>
inline Str operator+(const Str &s, const T &t) {
  std::stringstream ss;
  ss << s << t;
  return ss.str();
}

//template<typename T>
//inline Str operator+(const T &t, const Str &s) {
//  return {s << t};
//}

inline Str operator<<(const char *s, const Str &str) {
  return {str.str() + s};
}

//inline Str operator<<(const std::string &s, const Str &str) {
//  return {str.str() + s};
//}

inline Str operator<<(const Str &str, const char *s) {
  return {str.str() + s};
}

//inline Str operator<<(const Str &str, const std::string &s) {
//  return {str.str() + s};
//}

//template<typename T, std::enable_if_t<std::is_same_v<T, std::string> == false>>
template<typename T>
inline Str operator<<(const Str &s, T t) {
  std::stringstream ss;
  ss << t;
  return {s + ss.str()};
}

template<typename T, std::enable_if_t<std::is_same_v<T, std::string> == false>>
inline Str operator<<(T t, const Str &s) {
  std::stringstream ss;
  ss << t;
  return {s + ss.str()};
}

}

#endif //PONOS_PONOS_PONOS_COMMON_STR_H
