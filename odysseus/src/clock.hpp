#ifndef ODYSSEUS_CLOCK_H
#define ODYSSEUS_CLOCK_H

#include <chrono>
#include <ctime>

namespace odysseus {

	/* timer
	 * Helper class to measure time.
	 */
	class Timer {
		public:
			/* Default constructor.
			 * Starts the timer.
			 */
			Timer() {
				tick();
			}
			/* tick
			 * mark current time
			 */
			void tick() {
				lastTick = std::chrono::high_resolution_clock::now();
			}
			/* get
			 * @return elapsed time since last call to **tick**
			 */
			double tack() {
				auto curTick = std::chrono::high_resolution_clock::now();
				return std::chrono::duration<double, std::milli>(curTick - lastTick).count();
			}
			/* get
			 * same as calling **tack** first and then **tick**
			 * @return elapsed time since last call to **tick**
			 */
			double tackTick() {
				double elapsed = tack();
				tick();
				return elapsed;
			}

		private:
			std::chrono::high_resolution_clock::time_point lastTick;
	};

	/* clock
	 * Helper class to measure time.
	 */
	class Clock {
		public:
			/* Constructor.
			 * Starts the clock.
			 * @startTime **[in | optional]** starting time (**0.0**)
			 * @idealFrameInterval **[in | optional]** ideal frame interval (**30FPS**)
			 */
			explicit Clock(double startTime = 0.0, double idealFrameInterval = 1.0 / 30.0) {
				scale = 1.f;
				paused = false;
				time = startTime;
				ideal = idealFrameInterval;
			}
			/* update
			 * @dt **[in]**
			 */
			void update(double dt) {
				if(!paused) {
					double sdt = dt * scale;
					time += sdt;
				}
			}
			/* update
			 * updates clock using ideal frame interval
			 */
			void singleStep() {
				if(!paused) {
					double sdt = ideal * scale;
					time += sdt;
				}
			}
			/* pause
			 * pause clock
			 */
			void pause() {
				paused = true;
			}
			/* start
			 * starts clock
			 */
			void start() {
				paused = false;
			}
			/* set
			 * @dt **[in]** ideal frame interval
			 */
			void setIdealFrameInterval(double dt) {
				ideal = dt;
			}
			/* get
			 * @return **true** if clock is paused
			 */
			bool isPaused() const {
				return paused;
			}

		private:
			bool paused;
			float scale;
			double time;
			double ideal;
	};


} // odysseus namespace

#endif // ODYSSEUS_CLOCK_H
