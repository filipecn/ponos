#ifndef PONOS_ALGORITHM_INSERTION_SORT_H
#define PONOS_ALGORITHM_INSERTION_SORT_H

#include <functional>

namespace ponos {

	template<typename T>
		bool is_smaller(const T& a, const T& b) {
			return a < b;
		}

	template<typename T>
		void insertion_sort(T* v, size_t size, std::function<bool(const T& a, const T& b)> f = is_smaller<T>, size_t offset = 0) {
			size_t end = offset + size;
			for(size_t p = offset; p < end - 1; p++) {
				size_t m = p;
				for(size_t i = p + 1; i < end; i++)
					if(f(v[i], v[m]))
						m = i;
				if(m != p) {
					T tmp = v[p];
					v[p] = v[m];
					v[m] = tmp;
				}
			}
		}

} // ponos namespace

#endif // PONOS_ALGORITHM_INSERTION_SORT_H

