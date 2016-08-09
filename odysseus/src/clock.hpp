#pragma once

#include <chrono>
#include <ctime>

namespace odysseus {

	class Clock {
		public:
			Clock() {
				tick();
			}
			void tick() {
				lastTick = std::chrono::high_resolution_clock::now();
			}
			double tack() {
				auto curTick = std::chrono::high_resolution_clock::now();
				return std::chrono::duration<double, std::milli>(curTick - lastTick).count();
			}
			double tackTick() {
				double elapsed = tack();
				tick();
				return elapsed;
			}
		private:
			std::chrono::high_resolution_clock::time_point lastTick;
	};

} // odysseus namespace
