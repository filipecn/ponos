#pragma once

#include <aergia.h>
#include <hercules.h>

#include <thread>

class Demo {
	public:
		Demo(int w, int h)
			: width(w), height(h) {}
		void render() {
			static int i = 0;
			if(!i++) lastTick = clock.tackTick();
			update(clock.tackTick() - lastTick);
			lastTick = clock.tackTick();
			draw();
			double loopTime = clock.tack();
			// 60fps
			if(1000 / 60.0 > loopTime) {
				std::this_thread::sleep_for(
						std::chrono::milliseconds(
							static_cast<int>(
								(1000.0 / 60.0) - loopTime)));
			}
		}
		virtual void init() {};
	protected:
		virtual void update(double dt) {};
		virtual void draw() {};

		int width, height;
		ponos::Timer clock;
		double lastTick;
};
