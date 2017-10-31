#ifndef PONOS_SRC_COMMON_PARALLEL_H
#define PONOS_SRC_COMMON_PARALLEL_H

namespace ponos {
	/* mutex
	 * RWMutex provides reader-writer mutexes, which allow
	 * multiple threads make read operations but only one
	 * at a time to write.
	 */
	class RWMutex {
  	public:
			static RWMutex *create();
			static void destroy(RWMutex *m);
		private:
	};

	enum RWMutexLockType { READ, WRITE };

	/* Helper structure that acquires and releases reader-writer mutexes.
	 */
	struct RWMutexLock {
		/*@m <RWMutex> reference.
		 *@t Type of the lock: read-only(<READ>) or write(<WRITE>).
		 */
		RWMutexLock(RWMutex &m, RWMutexLock t);
		~RWMutexLock();
		void upgradeToWrite();
		void downgradeToRead();

		private:
		RWMutexLockType type;
		RWMutex &mutex;
	};

} // ponos namespace

#endif // PONOS_SRC_COMMON_PARALLEL_H

