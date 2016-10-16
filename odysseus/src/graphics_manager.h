#ifndef ODYSSEUS_GRAPHICS_MANAGER
#define ODYSSEUS_GRAPHICS_MANAGER

#include <aergia.h>

namespace odysseus {

	/* singleton
	 * Manages graphics
	 */
	class GraphicsManager {
  	public:
			static GraphicsManager &instance() {
				return instance_;
			}
			virtual ~GraphicsManager() {}

		private:
			GraphicsManager();
			GraphicsManager(GraphicsManager const&) = delete;
			void operator=(GraphicsManager const&) = delete;

			static GraphicsManager instance_;
	};

} // odysseus namespace

#endif // ODYSSEUS_GRAPHICS_MANAGER

