#ifndef PONOS_STRUCTURES_C_REGULAR_GRID_H
#define PONOS_STRUCTURES_C_REGULAR_GRID_H

#include "common/macros.h"
#include "log/debug.h"
#include "structures/c_grid_interface.h"

#include <algorithm>
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
				CRegularGrid() {}
				/* Constructor
				 * @d **[in]** dimensions
				 * @b **[in]** background (default value)
				 */
				CRegularGrid(const ivec3& d, const T& b, const vec3 cellSize = vec3(1.f), const vec3& offset = vec3());
				CRegularGrid(const ivec3& d, const T& b, const BBox& bb);

				~CRegularGrid();
				/* @inherit */
				void set(const ivec3& i, const T& v) override;
				void setAll(T v);
				/* @inherit */
				T operator()(const ivec3& i) const override {
					CHECK_IN_BETWEEN(i[0], 0, this->dimensions[0]);
					CHECK_IN_BETWEEN(i[1], 0, this->dimensions[1]);
					CHECK_IN_BETWEEN(i[2], 0, this->dimensions[2]);
					return data[i[0]][i[1]][i[2]];
				}
				/* @inherit */
				T& operator()(const ivec3& i) override {
					CHECK_IN_BETWEEN(i[0], 0, this->dimensions[0]);
					CHECK_IN_BETWEEN(i[1], 0, this->dimensions[1]);
					CHECK_IN_BETWEEN(i[2], 0, this->dimensions[2]);
					return data[i[0]][i[1]][i[2]];
				}
				/* @inherit */
				T operator()(const uint& i, const uint&j, const uint& k) const override {
					CHECK_IN_BETWEEN(static_cast<int>(i), 0, this->dimensions[0]);
					CHECK_IN_BETWEEN(static_cast<int>(j), 0, this->dimensions[1]);
					CHECK_IN_BETWEEN(static_cast<int>(k), 0, this->dimensions[2]);
					return data[i][j][k];
				}
				/* @inherit */
				T& operator()(const uint& i, const uint&j, const uint& k) override {
					CHECK_IN_BETWEEN(static_cast<int>(i), 0, this->dimensions[0]);
					CHECK_IN_BETWEEN(static_cast<int>(j), 0, this->dimensions[1]);
					CHECK_IN_BETWEEN(static_cast<int>(k), 0, this->dimensions[2]);
					return data[i][j][k];
				}
				T safeData(int i, int j, int k) const;
				T operator()(const float& x, const float&y, const float& z) const override {
					Point3 gp = this->toGrid(ponos::Point3(x, y, z));
					float p[3] = { gp.x, gp.y, gp.z };
					return trilinearInterpolate<T>(p, data, this->background, this->dimensions.v);
				}
				T operator()(const vec3& i) const override {
					return (*this)(i[0], i[1], i[2]);
				}
				void normalize() override;
				void normalizeElements() override;


			private:
				T*** data;
		};

	template<typename T>
		CRegularGrid<T>::CRegularGrid(const ivec3& d, const T& b, const vec3 cellSize, const vec3& offset) {
			this->dimensions = d;
			this->background = b;
			data = new T**[d[0]];
			for (int i = 0; i < d[0]; i++)
				data[i] = new T*[d[1]];
			for (int i = 0; i < d[0]; i++)
				for (int j = 0; j < d[1]; j++)
					data[i][j] = new T[d[2]];
			this->toWorld.reset();
			this->toWorld.scale(cellSize.x, cellSize.y, cellSize.z);
			this->toWorld.translate(offset);
			this->toWorld.computeInverse();
			this->toGrid = inverse(this->toWorld);
		}

	template<typename T>
		CRegularGrid<T>::CRegularGrid(const ivec3& d, const T& b, const BBox& bb) {
			this->dimensions = d;
			this->background = b;
			data = new T**[d[0]];
			for (int i = 0; i < d[0]; i++)
				data[i] = new T*[d[1]];
			for (int i = 0; i < d[0]; i++)
				for (int j = 0; j < d[1]; j++)
					data[i][j] = new T[d[2]];
			this->toWorld.reset();
			vec3 s = vec3(bb.size(0) / (d[0]), bb.size(1) / (d[1]), bb.size(2) / (d[2]));
			this->toWorld =
				translate(vec3(bb.pMin) + s * 0.5f) *
				scale(s[0], s[1], s[2]);

			//this->toWorld.scale(s[0], s[1], s[2]);
			//this->toWorld.translate(vec3(bb.pMin) + s * 0.5f + vec3(0, 0, 1));
			this->toWorld.computeInverse();
			this->toGrid = inverse(this->toWorld);

		}

	template<typename T>
		CRegularGrid<T>::~CRegularGrid() {
			for (int i = 0; i < this->dimensions[0]; i++)
				for (int j = 0; j < this->dimensions[1]; j++)
					delete[] data[i][j];
			for (int i = 0; i < this->dimensions[0]; i++)
				delete[] data[i];
			delete[] data;
		}

	template<typename T>
		void CRegularGrid<T>::setAll(T v){
			ivec3 ijk;
			FOR_INDICES0_3D(this->dimensions, ijk)
				data[ijk[0]][ijk[1]][ijk[2]] = v;
		}

	template<typename T>
		T CRegularGrid<T>::safeData(int i, int j, int k) const{
			return data
				[max(0, min(this->dimensions[0]-1,i))]
				[max(0, min(this->dimensions[1]-1,j))]
					[max(0, min(this->dimensions[2]-1,k))];
		}

	template<typename T>
		void CRegularGrid<T>::set(const ivec3& i, const T& v) {
			this->data[std::max(0, std::min(this->dimensions[0]-1,i[0]))][std::max(0, std::min(this->dimensions[1]-1,i[1]))][std::max(0, std::min(this->dimensions[2] - 1, i[2]))] = v;
		}

	template<typename T>
		void CRegularGrid<T>::normalize() {
			ivec3 ijk;
			T M = data[0][0][0];
			FOR_INDICES0_3D(this->dimensions, ijk)
				M = ponos::max(M, data[ijk[0]][ijk[1]][ijk[2]]);
			FOR_INDICES0_3D(this->dimensions, ijk)
				data[ijk[0]][ijk[1]][ijk[2]] /= M;
		}

	template<typename T>
		void CRegularGrid<T>::normalizeElements() {
			ivec3 ijk;
			FOR_INDICES0_3D(this->dimensions, ijk)
				data[ijk[0]][ijk[1]][ijk[2]] = ponos::normalize(data[ijk[0]][ijk[1]][ijk[2]]);

		}

	template<class T>
		class CRegularGrid2D : public CGrid2DInterface<T> {
			public:
				CRegularGrid2D();
				CRegularGrid2D(uint32_t w, uint32_t h);

				virtual ~CRegularGrid2D();

				T& operator() (int i, int j);
				T operator() (int i, int j) const;
				T safeData(int i, int j) const;

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
