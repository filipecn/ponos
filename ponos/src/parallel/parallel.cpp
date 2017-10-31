#include "parallel/parallel.h"

#include "common/defs.h"

#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>

namespace ponos {

void parallel_for(size_t start, size_t end,
                  std::function<void(size_t first, size_t last)> f, int gs) {
  if (start >= end)
    return;
  const uint length = end - start;
  const uint min_per_thread = gs <= 0 ? 25 : gs;
  const uint hardware_threads = std::thread::hardware_concurrency();
  const uint max_threads = (length + min_per_thread - 1) / min_per_thread;
  const uint num_threads =
      std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);
  const uint block_size = length / num_threads;
  std::vector<std::thread> threads(num_threads - 1);
  size_t first = 0;
  for (size_t i = 0; i < num_threads - 1; i++) {
    size_t last = std::min(first + block_size, static_cast<size_t>(length - 1));
    threads[i] = std::thread(f, start + first, start + last);
    first = last + 1;
  }
  f(start + first, end - 1);
  std::for_each(threads.begin(), threads.end(),
                std::mem_fn(&std::thread::join));
}

} // ponos namespace
