#include <cmath>
#include <iostream>

#define LOG_LOCATION \
  "[" << __FILE__ << ", line " << __LINE__ << "]"

#define ASSERT(A) \
  { \
    if(!(A)) \
      std::cout << "Assertion failed in " << LOG_LOCATION << std::endl; \
  } \

#define CHECK_IN_BETWEEN(A, B, C) \
  { \
    if(!((A) >= (B) && (A) <= (C))) \
      std::cout << "Assertion failed in " << LOG_LOCATION << std::endl; \
  } \

#define CHECK_FLOAT_EQUAL(A, B) \
  { \
    if(fabs((A) - (B)) < 1e-8) \
      std::cout << LOG_LOCATION << " " << std::endl; \
  } \

#define LOG \
  std::cout << LOG_LOCATION << " "

#define PRINT(A) \
  std::cout << A << std::endl;
