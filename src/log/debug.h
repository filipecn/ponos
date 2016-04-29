#include <cmath>
#include <iostream>

#define LOG_LOCATION \
  "[" << __FILE__ << ", line " << __LINE__ << "]"

#define ASSERT(A) \
  ((A) == true)? : std::cout << "Assertion failed in " << LOG_LOCATION << std::endl

#define CHECK_FLOAT_EQUAL(A, B) \
    (fabs((A) - (B)) < 1e-8)? : std::cout << LOG_LOCATION << " "
