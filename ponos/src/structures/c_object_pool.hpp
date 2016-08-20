#ifndef PONOS_STRUCTURES_C_OBJECT_POOL_H
#define PONOS_STRUCTURES_C_OBJECT_POOL_H

#include "common/defs.h"

#include <vector>

namespace ponos {
/*
	template<class T>
		class COPPtr {
			friend class CObjectPool;
			public:
			T * operator -> () const {
				return reinterpret_cast<T *>(pointerTable[tableIndex]);
			}
		};
*/
	template <class T, uint S>
	class CObjectPool {
  	public:
	 		CObjectPool();

			T* create();
			void destroy(T* object);

		private:
			T pool[S];
			uint activeObjects;
	};

} // ponos namespace

#endif
