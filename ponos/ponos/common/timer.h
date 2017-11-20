#ifndef PONOS_COMMON_TIMER_H
#define PONOS_COMMON_TIMER_H

#include <chrono>
#include <ctime>

namespace ponos {

/** \brief Helper class to measure time.
 */
class Timer {
public:
  /** \brief Default constructor.
   * Starts the timer.
   */
  Timer() { tick(); }
  /** \brief  tick
   *  mark current time
   */
  void tick() { lastTick = std::chrono::high_resolution_clock::now(); }
  /** \brief  get
   *  \return elapsed time since last call to **tick** in milliseconds
   */
  double tack() {
    auto curTick = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(curTick - lastTick)
        .count();
  }
  /** \brief  get
   * same as calling **tack** first and then **tick**
   * \return elapsed time since last call to **tick**
   */
  double tackTick() {
    double elapsed = tack();
    tick();
    return elapsed;
  }

private:
  std::chrono::high_resolution_clock::time_point lastTick;
};

} // ponos namespace

#endif // PONOS_COMMON_TIMER_H
