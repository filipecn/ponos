/*
 * Copyright (c) 2017 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
*/

#include <ponos/common/file.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>

#ifdef _WIN32
#include <fstream>
#include <string>
#include <windows.h>
// Allow use of freopen() without compilation warnings/errors.
// For use with custom allocated console window.
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#else
#include <unistd.h>
#endif

namespace ponos {
#ifdef WIN32
// TODO handle errors
int readFile(const char *filename, char **text) {
  std::ifstream file(filename);
  std::string str;
  std::string contents;
  while (std::getline(file, str)) {
    contents += str;
    contents.push_back('\n');
  }
  if (!contents.size())
    return 0;
  *text = new char[contents.size() + 1];
  std::strcpy(*text, contents.c_str());
  (*text)[contents.size()] = '\0';
  return contents.size();
}
#endif
#ifndef WIN32
int readFile(const char *filename, char **text) {
  int count;

  int fd = open(filename, O_RDONLY);
  if (fd == -1)
    return 0;

  size_t size = (size_t)(lseek(fd, 0, SEEK_END) + 1);
  close(fd);
  *text = new char[size];

  FILE *f = fopen(filename, "r");
  if (!f)
    return 0;

  fseek(f, 0, SEEK_SET);
  count = (int)fread(*text, 1, size, f);
  (*text)[count] = '\0';

  if (ferror(f))
    count = 0;

  fclose(f);
  return count;
}
#endif
} // ponos namespace
