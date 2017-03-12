#ifndef PONOS_ALGORITHM_SEARCH_H
#define PONOS_ALGORITHM_SEARCH_H

#include <iostream>
#include <functional>

namespace ponos {

template <typename T, typename V = T> int compare(const T &a, const V &v) {
  if (a > v)
    return 1;
  if (a < v)
    return -1;
  return 0;
}

template <typename T, typename V = T>
int binary_search(T *v, size_t size, const V &value,
                  std::function<int(const T &a, const V &v)> f = compare<T, V>,
                  size_t offset = 0) {
  size_t end = offset + size;
  int L = offset, H = end - 1;
  while (L + 1 < H) {
    int M = (L + H) >> 1;
    int cmp = f(v[M], value);
    if (cmp >= 0)
      H = M;
    else
      L = M;
  }
  if (f(v[H], value) == 0)
    return H;
  if (f(v[L], value) == 0)
    return L;
  return -1;
}

template <typename T, typename V = T>
int lower_bound(T *v, size_t size, const V &value,
                std::function<int(const T &a, const V &v)> f = compare<T, V>,
                size_t offset = 0) {
  size_t end = offset + size;
  int L = offset, H = end - 1;
  while (L + 1 < H) {
    int M = (L + H) >> 1;
    int cmp = f(v[M], value);
    if (cmp >= 0)
      H = M;
    else
      L = M;
  }
  if (f(v[H], value) < 0)
    return H;
  if (f(v[L], value) < 0)
    return L;
  return L - 1;
}

template <typename T, typename V = T>
int upper_bound(T *v, size_t size, const V &value,
                std::function<int(const T &a, const V &v)> f = compare<T, V>,
                size_t offset = 0) {
  size_t end = offset + size;
  int L = offset, H = end - 1;
  while (L + 1 < H) {
    int M = (L + H) >> 1;
    int cmp = f(v[M], value);
    if (cmp > 0)
      H = M;
    else
      L = M;
  }
  if (f(v[L], value) > 0)
    return L;
  if (f(v[H], value) > 0)
    return H;
  return H + 1;
}

} // ponos namespace

#endif // PONOS_ALGORITHM_SEARCH_H
