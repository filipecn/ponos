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

#define DUMP_VECTOR(V) \
  { \
    std::cout << "VECTOR in " << LOG_LOCATION << std::endl; \
    for (size_t i = 0; i < V.size(); ++i) \
      std::cout << V[i] << " "; \
    std::cout << std::endl; \
  } \

#define DUMP_MATRIX(M) \
  { \
    std::cout << "MATRIX in " << LOG_LOCATION << std::endl; \
    for (int i = 0; i < M.size(); ++i) { \
      for (int j = 0; j < M[i].size(); ++j) \
        std::cout << M[i][j] << " "; \
      } std::cout << std::endl; \
  }
