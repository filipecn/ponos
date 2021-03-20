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
///\file console_colors.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-03-19
///
///\brief

#ifndef PONOS_PONOS_PONOS_LOG_CONSOLE_COLORS_H
#define PONOS_PONOS_PONOS_LOG_CONSOLE_COLORS_H

#include <ponos/common/defs.h>
#include <string>

namespace ponos {

class ConsoleColors {
public:
  // SET
  static char bold[5];
  static char dim[5];
  static char underlined[5];
  static char blink[5];
  static char inverted[5];
  static char hidden[5];
  // RESET
  static char reset[5];
  static char reset_bold[6];
  static char reset_dim[6];
  static char reset_underlined[6];
  static char reset_blink[6];
  static char reset_inverted[6];
  static char reset_hidden[6];
  // 8/16 Colors
  static char default_color[6];
  static char black[6];
  static char red[6];
  static char green[6];
  static char yellow[6];
  static char blue[6];
  static char magenta[6];
  static char cyan[6];
  static char light_gray[6];
  static char dark_gray[6];
  static char light_red[6];
  static char light_green[6];
  static char light_yellow[6];
  static char light_blue[6];
  static char light_magenta[6];
  static char light_cyan[6];
  static char white[6];
  static char background_default_color[6];
  static char background_black[6];
  static char background_red[6];
  static char background_green[6];
  static char background_yellow[6];
  static char background_blue[6];
  static char background_magenta[6];
  static char background_cyan[6];
  static char background_light_gray[6];
  static char background_dark_gray[7];
  static char background_light_red[7];
  static char background_light_green[7];
  static char background_light_yellow[7];
  static char background_light_blue[7];
  static char background_light_magenta[7];
  static char background_light_cyan[7];
  static char background_white[7];
  // 88/256 Colors
  inline static std::string color(u8 color_number) {
    return std::string("\e[38;5;") + std::to_string(color_number) + "m";
  }
  inline static std::string background_color(u8 color_number) {
    return std::string("\e[48;5;") + std::to_string(color_number) + "m";
  }
};

}

#endif //PONOS_PONOS_PONOS_LOG_CONSOLE_COLORS_H
