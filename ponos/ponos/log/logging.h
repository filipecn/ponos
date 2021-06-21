//
// Created by filipecn on 20/06/2021.
//


/// Copyright (c) 2021, FilipeCN.
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
///\file logging.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-06-20
///
///\brief

#ifndef PONOS_PONOS_PONOS_LOG_LOGGING_H
#define PONOS_PONOS_PONOS_LOG_LOGGING_H

#include <ponos/common/str.h>
#include <ponos/log/console_colors.h>
#include <cstdarg>
#include <chrono>

namespace ponos {

class Log {
public:
  /// \return
  static inline Str label() {
    const std::chrono::time_point<std::chrono::system_clock> now =
        std::chrono::system_clock::now();
    const std::time_t t_c = std::chrono::system_clock::to_time_t(now);
    Str s;
    s += std::put_time(std::localtime(&t_c), "[%F %T] ");
    return s;
  }
  ///
  /// \tparam Ts
  /// \param fmt
  /// \param args
  template<typename ...Ts>
  static inline void info(const std::string &fmt, Ts &&...args) {
    Str s;
    if (use_colors)
      s += ConsoleColors::color(info_label_color);
    if (use_colors)
      s += label() << " [info] ";
    if (use_colors)
      s += ConsoleColors::color(info_color);
    s += Str::format(fmt, std::forward<Ts>(args)...);
    if (use_colors)
      s += ConsoleColors::reset;
    printf("%s\n", s.c_str());
  }
  ///
  /// \tparam Ts
  /// \param fmt
  /// \param args
  template<typename ...Ts>
  static inline void warn(const std::string &fmt, Ts &&...args) {
    Str s;
    if (use_colors)
      s += ConsoleColors::color(warn_label_color);
    if (use_colors)
      s += label() << " [warn] ";
    if (use_colors)
      s += ConsoleColors::color(warn_color);
    s += Str::format(fmt, std::forward<Ts>(args)...);
    if (use_colors)
      s += ConsoleColors::reset;
    printf("%s\n", s.c_str());
  }
  ///
  /// \tparam Ts
  /// \param fmt
  /// \param args
  template<typename ...Ts>
  static inline void error(const std::string &fmt, Ts &&...args) {
    Str s;
    if (use_colors)
      s += ConsoleColors::color(error_label_color);
    if (use_colors)
      s += label() << " [error] ";
    if (use_colors)
      s += ConsoleColors::color(error_color);
    s += Str::format(fmt, std::forward<Ts>(args)...);
    if (use_colors)
      s += ConsoleColors::reset;
    printf("%s\n", s.c_str());
  }
  ///
  /// \tparam Ts
  /// \param fmt
  /// \param args
  template<typename ...Ts>
  static inline void critical(const std::string &fmt, Ts &&...args) {
    Str s;
    if (use_colors)
      s += ConsoleColors::color(critical_label_color);
    if (use_colors)
      s += label() << " [critical] ";
    if (use_colors)
      s += ConsoleColors::color(critical_color);
    s += Str::format(fmt, std::forward<Ts>(args)...);
    if (use_colors)
      s += ConsoleColors::reset;
    printf("%s\n", s.c_str());
  }

  static bool use_colors;

  static u8 info_color;
  static u8 warn_color;
  static u8 error_color;
  static u8 critical_color;

  static u8 info_label_color;
  static u8 warn_label_color;
  static u8 error_label_color;
  static u8 critical_label_color;
};

}

#endif //PONOS_PONOS_PONOS_LOG_LOGGING_H
