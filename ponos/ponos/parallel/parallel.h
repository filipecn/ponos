#ifndef PONOS_PARALLEL_H
#define PONOS_PARALLEL_H

#include <functional>

namespace ponos {

/* \brief loop
 * \param start **[in]**
 * \param end **[in]**
 * \param f **[in]**
 * \param gs **[in | optional]** grain size
 * Breaks the interval **[start,end)** into chunks and call **f** for each chunk
 * in parallel. The size
 * of each chunk is given by **gs** (or is automatically computed if not given).
 */
void parallel_for(size_t start, size_t end,
                  std::function<void(size_t first, size_t last)> f,
                  int gs = -1);

/* \brief query
 * \param start **[in]** first element index
 * \param end **[in]** last + 1 element index
 * \param array **[in]** array of elements
 * \param gs **[in | optional]** grain size
 *
 * \return the index of the first element with maximum value in the array
 */
template <typename T>
size_t parallel_max(size_t start, size_t end, const T *array, int gs = -1) {
  /*		if(start >= end)
                          return;
                  const uint length = end - start;
                  const uint min_per_thread = gs <= 0 ? 25 : gs;
                  const uint hardware_threads =
     std::thread::hardware_concurrency();
                  const uint max_threads = (length + min_per_thread - 1) /
     min_per_thread;
                  const uint num_threads = std::min(hardware_threads != 0 ?
     hardware_threads : 2, max_threads);
                  const uint block_size = length / num_threads;
                  std::vector<std::thread> threads(num_threads-1);
                  size_t first = 0;
                  for(size_t i = 0; i < num_threads - 1; i++) {
                          size_t last = std::min(first + block_size,
     static_cast<size_t>(length - 1));
                          threads[i] = std::thread(f, first, last);
                          first = last + 1;
                  }
                  f(first, end - 1);
                  std::for_each(threads.begin(), threads.end(),
     std::mem_fn(&std::thread::join));
*/
  return 0;
}
} // ponos namespace

#endif // PONOS_PARALLEL_H
