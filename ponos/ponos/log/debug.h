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

#ifndef PONOS_LOG_DEBUG_H
#define PONOS_LOG_DEBUG_H

#include <cmath>
#include <iostream>
#include <ponos/common/defs.h>
#include <sstream>

#ifndef UNUSED_VARIABLE
#define UNUSED_VARIABLE(x) ((void)x)
#endif
#ifndef LOG_LOCATION
#define LOG_LOCATION "[" << __FILE__ << "][" << __LINE__ << "]"
#endif
#ifndef FATAL_ASSERT
#define FATAL_ASSERT(A)                                                        \
  {                                                                            \
    if (!(A)) {                                                                \
      std::cout << "Assertion failed in " << LOG_LOCATION << std::endl;        \
      exit(1);                                                                 \
    }                                                                          \
  }
#endif
#ifndef ASSERT
#define ASSERT(A)                                                              \
  {                                                                            \
    if (!(A))                                                                  \
      std::cout << "Assertion failed in " << LOG_LOCATION << std::endl;        \
  }
#endif

#ifndef ASSERT_MESSAGE
#define ASSERT_MESSAGE(A, B)                                                   \
  {                                                                            \
    if (!(A))                                                                  \
      std::cerr << "Assertion failed in " << LOG_LOCATION << std::endl         \
                << B << std::endl;                                             \
  }
#endif
#ifndef ASSERT_EQ
#define ASSERT_EQ(A, B)                                                        \
  {                                                                            \
    if ((A) != (B)) {                                                          \
      std::cout << "Assertion failed in " << LOG_LOCATION << std::endl;        \
      std::cout << (A) << " != " << (B) << std::endl;                          \
    }                                                                          \
  }
#endif
#ifndef CHECK_IN_BETWEEN
#define CHECK_IN_BETWEEN(A, B, C)                                              \
  {                                                                            \
    if (!((A) >= (B) && (A) <= (C)))                                           \
      std::cout << "Assertion failed in " << LOG_LOCATION << std::endl;        \
  }
#endif
#ifndef CHECK_FLOAT_EQUAL
#define CHECK_FLOAT_EQUAL(A, B)                                                \
  {                                                                            \
    if (fabs((A) - (B)) < 1e-8)                                                \
      std::cout << LOG_LOCATION << " " << std::endl;                           \
  }
#endif

#ifndef PING
#define PING std ::cerr << LOG_LOCATION << "[" << __FUNCTION__ << "]: PING\n";
#endif

#ifndef LOG
#define LOG std::cout << LOG_LOCATION << " "
#endif
#ifndef PRINT
#define PRINT(A) std::cout << A << std::endl;
#endif
#ifndef DUMP_VECTOR
#define DUMP_VECTOR(V)                                                         \
  {                                                                            \
    std::cout << "VECTOR in " << LOG_LOCATION << std::endl;                    \
    for (size_t i = 0; i < V.size(); ++i)                                      \
      std::cout << V[i] << " ";                                                \
    std::cout << std::endl;                                                    \
  }
#endif
#ifndef DUMP_MATRIX
#define DUMP_MATRIX(M)                                                         \
  {                                                                            \
    std::cout << "MATRIX in " << LOG_LOCATION << std::endl;                    \
    for (int i = 0; i < M.size(); ++i) {                                       \
      for (int j = 0; j < M[i].size(); ++j)                                    \
        std::cout << M[i][j] << " ";                                           \
    }                                                                          \
    std::cout << std::endl;                                                    \
  }
#endif

namespace ponos {

// Condition to stop processing
inline void concatenate(std::ostringstream &s) { UNUSED_VARIABLE(s); }

template <typename H, typename... T>
void concatenate(std::ostringstream &s, H p, T... t) {
  s << p;
  concatenate(s, t...);
}

template <class... Args> std::string concat(const Args &... args) {
  std::ostringstream s;
  concatenate(s, args...);
  return s.str();
}

inline void printBits(u32 n) {
  for (int i = 31; i >= 0; i--)
    if ((1 << i) & n)
      std::cout << '1';
    else
      std::cout << '0';
}
} // namespace ponos

#endif
