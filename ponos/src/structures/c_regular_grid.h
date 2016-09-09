#ifndef PONOS_STRUCTURES_C_REGULAR_GRID_H
#define PONOS_STRUCTURES_C_REGULAR_GRID_H

#include "log/debug.h"
#include "structures/c_grid_interface.h"

#include <memory>
#include <vector>

namespace ponos {

	/* grid
	 *
	 * Simple matrix structure.
	 */
	template<class T = float>
		class CRegularGrid : public CGridInterface<T> {
			public:
				/* Constructor
				 * @d **[in]** dimensions
				 * @b **[in]** background (default value)
				 */
				CRegularGrid(const ivec3& d, const T& b);
				~CRegularGrid();
				/* @inherit */
				void set(const ivec3& i, const T& v) override {}
				/* @inherit */
				T operator()(const ivec3& i) const override {}
				/* @inherit */
				T& operator()(const ivec3& i) override {}
				/* @inherit */
				T operator()(const uint& i, const uint&j, const uint& k) const override {}
				/* @inherit */
				T& operator()(const uint& i, const uint&j, const uint& k) override {}

			private:
				T*** data;
		};

	template<class T>
		class CRegularGrid2D : public CGrid2DInterface<T> {
			public:
				CRegularGrid2D();
				CRegularGrid2D(uint32_t w, uint32_t h);

				virtual ~CRegularGrid2D();

				T& operator() (int i, int j) override;
				T operator() (int i, int j) const override;
				T safeData(int i, int j) const override;

				void set(uint32_t w, uint32_t h, Vector2 offset, Vector2 cellSize);
				void setAll(T v);

			private:
				std::vector<std::vector<T> >data;
		};

	template<typename T>
		CRegularGrid2D<T>::CRegularGrid2D(){
			this->width = this->height = 0;
			this->useBorder = false;
		}

	template<typename T>
		CRegularGrid2D<T>::CRegularGrid2D(uint32_t w, uint32_t h) {
			this->width = w;
			this->height = h;
		}

	template<typename T>
		CRegularGrid2D<T>::~CRegularGrid2D() {}

	template<typename T>
		void CRegularGrid2D<T>::set(uint32_t w, uint32_t h, Vector2 offset, Vector2 cellSize) {
			set(w, h);
			set(offset, cellSize);
			data.resize(w, std::vector<T>());
			for (int i = 0; i < w; i++)
				data[i].resize(h);
		}

	template<typename T>
		T& CRegularGrid2D<T>::operator() (int i, int j) {
			CHECK_IN_BETWEEN(i, 0, this->width);
			CHECK_IN_BETWEEN(j, 0, this->height);
			return data[i][j];
		}

	template<typename T>
		T CRegularGrid2D<T>::operator() (int i, int j) const {
			CHECK_IN_BETWEEN(i, 0, this->width);
			CHECK_IN_BETWEEN(j, 0, this->height);
			return data[i][j];
		}

	template<typename T>
		void CRegularGrid2D<T>::setAll(T v){
			for (int i = 0; i < this->width; i++)
				for (int j = 0; j < this->height; j++)
					data[i][j] = v;
		}

	template<typename T>
		T CRegularGrid2D<T>::safeData(int i, int j) const{
			return data[max(0, min(this->width-1,i))][max(0, min(this->height-1,j))];
		}

}  // ponos namespace

#endif // PONOS_STRUCTURES_C_REGULAR_GRID_H
