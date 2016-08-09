#pragma once

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
	template <class T, size_t S>
	class CObjectPool {
  	public:
	 		CObjectPool();

			T* create();
			void destroy(T* object);

		private:
			T pool[S];
			size_t activeObjects;
	};

} // ponos namespace

