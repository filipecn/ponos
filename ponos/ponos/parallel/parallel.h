#ifndef PONOS_PARALLEL_H
#define PONOS_PARALLEL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ponos {

class ParallelForLoop {
public:
  /// \param func1D cell back to 1D index loop
  /// \param maxIndex
  /// \param chunkSize
  /// \param profilerState
  ParallelForLoop(std::function<void(int)> func1D, int64_t maxIndex,
                  int chunkSize, int profilerState);

  [[nodiscard]] bool finished() const;

  std::function<void(int)> func1D;
  const int64_t maxIndex;
  const int chunkSize, profilerState;
  int64_t nextIndex = 0; //!< tracks the next loop index to be executed
  int activeWorkers = 0; //!< records how many worker threads are currently
  //!< running iterations of the loop
  ParallelForLoop *next = nullptr; //!< maintain the linked list of nested loops
};

/// Persistent Threads approach. Stores a pool of threads that are initialized
/// only once in the execution time.
class ThreadPool {
public:
  static void workerThreadFunc(int tIndex);
  static int numSystemCores();

  static std::vector<std::thread> threads;
  static bool shutdownThreads;
  static thread_local int threadIndex;
  static ParallelForLoop *workList;
  static std::mutex workListMutex;
  static std::condition_variable workListCondition;
};

class Parallel {
public:
  /// \param f callback(int index) called to each loop index iteration
  /// \param count defines the [0, count) index sequence
  /// \param chunkSize thread granularity
  static void loop(const std::function<void(int)> &f, int count, int chunkSize = 1);
  /// \param start **[in]**
  /// \param end **[in]**
  /// \param f **[in]**
  /// \param gs **[in | optional]** grain size
  /// Breaks the interval **[start,end)** into chunks and call **f** for each chunk
  /// in parallel. The size of each chunk is given by **gs** (or is automatically
  /// computed if not given).
  static void loop(size_t start, size_t end,
                   std::function<void(size_t first, size_t last)> f, int gs = -1);
  /// \param start **[in]** first element index
  /// \param end **[in]** last + 1 element index
  /// \param array **[in]** array of elements_
  /// \param gs **[in | optional]** grain size
  /// \return the index of the first element with maximum value in the array
  template<typename T>
  static size_t max(size_t start, size_t end, const T *array, int gs = -1) {
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
};

} // namespace ponos

#endif // PONOS_PARALLEL_H
