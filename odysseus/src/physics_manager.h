#ifndef ODYSSEUS_PHYSICS_MANAGER_H
#define ODYSSEUS_PHYSICS_MANAGER_H

#include "hercules.h"

namespace odysseus {

	/* singleton
	 * Manages physics
	 */
	class PhysicsManager {
  	public:
			static PhysicsManager &instance() {
				return instance_;
			}
			virtual ~PhysicsManager() {}

		private:
	 		PhysicsManager();
			PhysicsManager(PhysicsManager const&) = delete;
			void operator=(PhysicsManager const&) = delete;

			static PhysicsManager instance_;
	};

} // odysseus namespace

#endif // ODYSSEUS_PHYSICS_MANAGER_H

