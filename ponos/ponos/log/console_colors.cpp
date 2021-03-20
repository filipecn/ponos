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
///\file console_colors.cpp.c
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-03-19
///
///\brief

#include <ponos/log/console_colors.h>

namespace ponos {


char ConsoleColors::bold[5] = "\e[1m";
char ConsoleColors::dim[5] = "\e[2m";
char ConsoleColors::underlined[5] = "\e[4m";
char ConsoleColors::blink[5] = "\e[5m";
char ConsoleColors::inverted[5] = "\e[7m";
char ConsoleColors::hidden[5] = "\e[8m";

char ConsoleColors::reset[5] = "\e[0m";
char ConsoleColors::reset_bold[6] = "\e[21m";
char ConsoleColors::reset_dim[6] = "\e[22m";
char ConsoleColors::reset_underlined[6] = "\e[24m";
char ConsoleColors::reset_blink[6] = "\e[25m";
char ConsoleColors::reset_inverted[6] = "\e[27m";
char ConsoleColors::reset_hidden[6] = "\e[28m";

char ConsoleColors::default_color[6] = "\e[39m";
char ConsoleColors::black[6] = "\e[30m";
char ConsoleColors::red[6] = "\e[31m";
char ConsoleColors::green[6] = "\e[32m";
char ConsoleColors::yellow[6] = "\e[33m";
char ConsoleColors::blue[6] = "\e[34m";
char ConsoleColors::magenta[6] = "\e[35m";
char ConsoleColors::cyan[6] = "\e[36m";
char ConsoleColors::light_gray[6] = "\e[37m";
char ConsoleColors::dark_gray[6] = "\e[90m";
char ConsoleColors::light_red[6] = "\e[91m";
char ConsoleColors::light_green[6] = "\e[92m";
char ConsoleColors::light_yellow[6] = "\e[93m";
char ConsoleColors::light_blue[6] = "\e[94m";
char ConsoleColors::light_magenta[6] = "\e[95m";
char ConsoleColors::light_cyan[6] = "\e[96m";
char ConsoleColors::white[6] = "\e[97m";

char ConsoleColors::background_default_color[6] = "\e[49m";
char ConsoleColors::background_black[6] = "\e[40m";
char ConsoleColors::background_red[6] = "\e[41m";
char ConsoleColors::background_green[6] = "\e[42m";
char ConsoleColors::background_yellow[6] = "\e[43m";
char ConsoleColors::background_blue[6] = "\e[44m";
char ConsoleColors::background_magenta[6] = "\e[45m";
char ConsoleColors::background_cyan[6] = "\e[46m";
char ConsoleColors::background_light_gray[6] = "\e[47m";
char ConsoleColors::background_dark_gray[7] = "\e[100m";
char ConsoleColors::background_light_red[7] = "\e[101m";
char ConsoleColors::background_light_green[7] = "\e[102m";
char ConsoleColors::background_light_yellow[7] = "\e[103m";
char ConsoleColors::background_light_blue[7] = "\e[104m";
char ConsoleColors::background_light_magenta[7] = "\e[105m";
char ConsoleColors::background_light_cyan[7] = "\e[106m";
char ConsoleColors::background_white[7] = "\e[107m";

}