#ifndef ODYSSEUS_GAME_H
#define ODYSSEUS_GAME_H

#include <aergia.h>
#include <memory>

#include "clock.hpp"
#include <thread>

namespace odysseus {

	class Game {
		public:
			Game(const char* title, int width, int height) {}
			Game() {
				aergia::createGraphicsDisplay(800, 800, "Game");
				MS_PER_UPDATE = 15;
			}
			virtual int run() {
				aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
				init();
				Timer clock;
				double lag = 0.0;
				while(gd.isRunning()) {
					lag += clock.tackTick();
					gd.processInput();
					processInput();
					int k = 0;
					while(lag >= MS_PER_UPDATE && k < 100) {
						update();
						lag -= MS_PER_UPDATE;
						k++;
					}
					gd.beginFrame();
					render();
					gd.endFrame();
					double loopTime = clock.tack();
					// 60fps
					if(1000 / 60.0 > loopTime) {
						std::this_thread::sleep_for(
								std::chrono::milliseconds(
									static_cast<int>(
										(1000.0 / 60.0) - loopTime)));
					}
				}
				return 0;
			}

		protected:
			double MS_PER_UPDATE;
			virtual void init() = 0;
			virtual void update() = 0;
			virtual void render() = 0;
			virtual void processInput() = 0;
	};

} // odysseus namespace

#endif // ODYSSEUS_GAME_H
