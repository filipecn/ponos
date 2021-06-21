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
#include <ponos/log/logging.h>
#include <sstream>

#ifndef PONOS_DEBUG
#define PONOS_DEBUG
#endif

#ifndef CHECKS_ENABLED
#define CHECKS_ENABLED
#endif

#ifndef ASSERTIONS_ENABLED
#define ASSERTIONS_ENABLED
#endif

#ifndef INFO_ENABLED
#define INFO_ENABLED
#endif

/****************************************************************************
                                 UTILS
****************************************************************************/
#ifndef LOG_LOCATION
#define LOG_LOCATION "[" << __FILE__ << "][" << __LINE__ << "]"
#endif
/****************************************************************************
                          COMPILATION WARNINGS
****************************************************************************/
#ifndef PONOS_UNUSED_VARIABLE
#define PONOS_UNUSED_VARIABLE(x) ((void)x)
#endif
/****************************************************************************
                               DEBUG MODE
****************************************************************************/
#ifdef PONOS_DEBUG
#define PONOS_DEBUG_CODE(CODE_CONTENT) {CODE_CONTENT}
#else
#define PONOS_DEBUG_CODE(CODE_CONTENT)
#endif
/****************************************************************************
                                 INFO
****************************************************************************/
#ifdef INFO_ENABLED

#ifndef PONOS_PING
#define PONOS_PING Log::info("[{}][{}][{}]", __FILE__, __LINE__, __FUNCTION__);
#endif

#ifndef PONOS_LOG
#define PONOS_LOG(A) Log::info("[{}][{}][{}]: {}", __FILE__, __LINE__, __FUNCTION__, A);
#endif

#ifndef PONOS_LOG_WARNING
#define PONOS_LOG_WARNING(A) Log::warn("[{}][{}][{}]: {}", __FILE__, __LINE__, __FUNCTION__, A);
#endif

#ifndef PONOS_LOG_ERROR
#define PONOS_LOG_ERROR(A) Log::error("[{}][{}][{}]: {}", __FILE__, __LINE__, __FUNCTION__, A);
#endif

#ifndef PONOS_LOG_CRITICAL
#define PONOS_LOG_CRITICAL(A) Log::critical("[{}][{}][{}]: %s", __FILE__, __LINE__, __FUNCTION__, A);
#endif

#ifndef PONOS_LOG_VARIABLE
#define PONOS_LOG_VARIABLE(A) Log::info("[{}][{}][{}]: {} = {}", __FILE__, __LINE__, __FUNCTION__, #A, A);
#endif

#else

#define PONOS_PING
#define PONOS_LOG
#define PONOS_LOG_VARIABLE

#endif // CHECKS_ENABLED
/****************************************************************************
                                 CHECKS
****************************************************************************/
#ifdef CHECKS_ENABLED

#define PONOS_CHECK_EXP(expr) \
  if(expr) {}          \
  else {                      \
    Log::warn("[{}][{}][CHECK_EXP FAIL {}]", __FILE__, __LINE__, (#expr));\
  }

#define PONOS_CHECK_EXP_WITH_LOG(expr, M) \
  if(expr) {}          \
  else {                      \
    Log::warn("[{}][{}][CHECK_EXP FAIL {}]: {}", __FILE__, __LINE__, (#expr), M);\
  }
#else

#define PONOS_CHECK_EXP(expr)
#define PONOS_CHECK_EXP_WITH_LOG(expr, M)

#endif // CHECKS_ENABLED
/****************************************************************************
                             ASSERTION
****************************************************************************/
#ifdef ASSERTIONS_ENABLED

//#define debugBreak() asm ("int 3")
#define debugBreak()

#define PONOS_ASSERT(expr) \
  if(expr) {}          \
  else {                   \
    Log::error("[{}][{}][ASSERT FAIL {}]", __FILE__, __LINE__, #expr);\
    debugBreak();                                                                         \
  }
#define PONOS_ASSERT_WITH_LOG(expr, M) \
  if(expr) {}          \
  else {                   \
    Log::error("[{}][{}][ASSERT FAIL {}]: {}", __FILE__, __LINE__, #expr, M);\
    debugBreak();                                                                         \
  }
#else

#define PONOS_ASSERT(expr)
#define PONOS_ASSERT_WITH_LOG(expr, M)

#endif // ASSERTIONS_ENABLED
/****************************************************************************
                             CODE FLOW
****************************************************************************/
#define PONOS_RETURN_IF(A, R)                                      \
  if (A) {                                                             \
    return R;                                                             \
  }
#define PONOS_RETURN_IF_NOT(A, R)                                      \
  if (!(A)) {                                                             \
    return R;                                                             \
  }
#define PONOS_LOG_AND_RETURN_IF_NOT(A, R, M)                         \
  if (!(A)) {                                                         \
    PONOS_LOG(M)                                                                  \
    return R;                                                             \
  }

namespace ponos {

inline void printBits(u32 n) {
  for (int i = 31; i >= 0; i--)
    if ((1 << i) & n)
      std::cout << '1';
    else
      std::cout << '0';
}
} // namespace ponos

#endif
