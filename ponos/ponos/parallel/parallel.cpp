#include <ponos/common/defs.h>
#include <ponos/parallel/parallel.h>

#include <algorithm>
#include <iostream>

namespace ponos {

std::vector<std::thread> ThreadPool::threads;
bool ThreadPool::shutdownThreads = false;
thread_local int ThreadPool::threadIndex;
ParallelForLoop *ThreadPool::workList;
std::mutex ThreadPool::workListMutex;
std::condition_variable ThreadPool::workListCondition;

ParallelForLoop::ParallelForLoop(std::function<void(int)> func1D,
                                 int64_t maxIndex, int chunkSize,
                                 int profilerState)
    : func1D(std::move(func1D)), maxIndex(maxIndex), chunkSize(chunkSize),
      profilerState(profilerState) {}

bool ParallelForLoop::finished() const {
  return nextIndex >= maxIndex && activeWorkers == 0;
}

int ThreadPool::numSystemCores() {
  return std::max(1u, std::thread::hardware_concurrency());
}

void ThreadPool::workerThreadFunc(int tIndex) {
  threadIndex = tIndex;
  std::unique_lock<std::mutex> lock(workListMutex);
  while (!shutdownThreads) {
    if (!workList) {
      // sleep until there are more tasks to run
      workListCondition.wait(lock);
    } else {
      // get work from workList and run loop iterations
      ParallelForLoop &loop = *workList;
      // run a chunk of loop iterations for loop
      //    find the set of loop iterations to run next
      int64_t indexStart = loop.nextIndex;
      int64_t indexEnd = std::min(indexStart + loop.chunkSize, loop.maxIndex);
      //    update loop to reflect iterations this thread will run
      loop.nextIndex = indexEnd;
      if (loop.nextIndex == loop.maxIndex)
        ThreadPool::workList = loop.next;
      loop.activeWorkers++;
      //    run loop indices in [indexStart, indexEnd)
      lock.unlock();
      for (int index = indexStart; index < indexEnd; ++index) {
        if (loop.func1D)
          loop.func1D(index);
        // handle other types of loops TODO
      }
      lock.lock();
      //    update loop to reflect completion of iterations]
      loop.activeWorkers--;
      if (loop.finished())
        workListCondition.notify_all();
    }
  }
  // report thread statistics at worker thread exit
  // TODO reportThreadStats();
}

void parallel_for(const std::function<void(int)> &f, int count, int chunkSize) {
  // run iterations immediately if not using threads or if count is small
  if (count < chunkSize) {
    for (int i = 0; i < count; i++)
      f(i);
    return;
  }
  // launch worker threads if needed
  if (ThreadPool::threads.empty()) {
    ThreadPool::threadIndex = 0;
    for (int i = 0; i < ThreadPool::numSystemCores() - 1; ++i)
      ThreadPool::threads.push_back(
          std::thread(ThreadPool::workerThreadFunc, i + 1));
  }
  // create enqueue parallel_for_loop for this loop
  ParallelForLoop loop(f, count, chunkSize, 0 /*CurrentProfilerState() TODO*/);
  ThreadPool::workListMutex.lock();
  loop.next = ThreadPool::workList;
  ThreadPool::workList = &loop;
  ThreadPool::workListMutex.unlock();
  // notify worker threads of work to be one
  std::unique_lock<std::mutex> lock(ThreadPool::workListMutex);
  ThreadPool::workListCondition.notify_all();
  // help out with parallel loop iterations in the current thread
  while (!loop.finished()) {
    // run a chunk of loop iterations for loop
    //    find the set of loop iterations to run next
    int64_t indexStart = loop.nextIndex;
    int64_t indexEnd = std::min(indexStart + loop.chunkSize, loop.maxIndex);
    //    update loop to reflect iterations this thread will run
    loop.nextIndex = indexEnd;
    if (loop.nextIndex == loop.maxIndex)
      ThreadPool::workList = loop.next;
    loop.activeWorkers++;
    //    run loop indices in [indexStart, indexEnd)
    lock.unlock();
    for (int index = indexStart; index < indexEnd; ++index) {
      if (loop.func1D)
        loop.func1D(index);
      // handle other types of loops TODO
    }
    lock.lock();
    //    update loop to reflect completion of iterations]
    loop.activeWorkers--;
  }
}

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
  u32 first = 0;
  for (size_t i = 0; i < num_threads - 1; i++) {
    size_t last = std::min(first + block_size, static_cast<u32>(length - 1));
    threads[i] = std::thread(f, start + first, start + last);
    first = last + 1;
  }
  f(start + first, end - 1);
  std::for_each(threads.begin(), threads.end(),
                std::mem_fn(&std::thread::join));
}

} // namespace ponos
