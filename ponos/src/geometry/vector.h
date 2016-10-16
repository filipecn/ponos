#ifndef PONOS_GEOMETRY_VECTOR_H
#define PONOS_GEOMETRY_VECTOR_H

#include "geometry/normal.h"
#include "geometry/numeric.h"
#include "log/debug.h"

#include <cstring>

namespace ponos {

class Point2;
class Vector2 {
	public:
		Vector2();
		explicit Vector2(float _x, float _y);
		explicit Vector2(const Point2& p);
		// access
		float operator[](int i) const {
			ASSERT(i >= 0 && i <= 1);
			return (&x)[i];
		}
		float& operator[](int i) {
			ASSERT(i >= 0 && i <= 1);
			return (&x)[i];
		}
		// arithmetic
		Vector2 operator+(const Vector2& v) const {
			return Vector2(x + v.x, y + v.y);
		}
		Vector2& operator+=(const Vector2& v) {
			x += v.x;
			y += v.y;
			return *this;
		}
		Vector2 operator-(const Vector2& v) const {
			return Vector2(x - v.x, y - v.y);
		}
		Vector2& operator-=(const Vector2& v) {
			x -= v.x;
			y -= v.y;
			return *this;
		}
		Vector2 operator*(float f) const {
			return Vector2(x * f, y * f);
		}
		Vector2& operator*=(float f) {
			x *= f;
			y *= f;
			return *this;
		}
		Vector2 operator/(float f) const {
			CHECK_FLOAT_EQUAL(f, 0.f);
			float inv = 1.f / f;
			return Vector2(x * inv, y * inv);
		}
		Vector2& operator/=(float f) {
			CHECK_FLOAT_EQUAL(f, 0.f);
			float inv = 1.f / f;
			x *= inv;
			y *= inv;
			return *this;
		}
		Vector2 operator-() const {
			return Vector2(-x, -y);
		}
		// normalization
		float length2() const {
			return x * x + y * y;
		}
		float length() const {
			return sqrtf(length2());
		}
		bool HasNaNs() const;

		friend std::ostream& operator<<(std::ostream& os, const Vector2& v);
		float x, y;
};

inline Vector2 operator*(float f, const Vector2& v) {
	return v*f;
}

inline float dot(const Vector2& a, const Vector2& b) {
	return a.x * b.x + a.y * b.y;
}

inline Vector2 normalize(const Vector2& v) {
	return v / v.length();
}

inline Vector2 orthonormal(const Vector2& v, bool first = true) {
	Vector2 n = normalize(v);
	if(first)
		return Vector2(-n.y, n.x);
	return Vector2(n.y, -n.x);
}

inline float cross(const Vector2& a, const Vector2& b) {
	return a.x * b.y - a.y * b.x;
}

class Point3;
class Vector3 {
	public:
		Vector3();
		explicit Vector3(float _f);
		explicit Vector3(float _x, float _y, float _z);
		explicit Vector3(const Normal& n);
		explicit Vector3(const Point3& p);
		// boolean
		bool operator==(const Vector3& v) {
			return IS_EQUAL(x, v.x) && IS_EQUAL(y, v.y) && IS_EQUAL(z, v.z);
		}
		bool operator<(const Vector3& v) {
			if(x < v.x) return true;
			if(y < v.y) return true;
			if(z < v.z) return true;
			return false;
		}
		bool operator>(const Vector3& v) {
			if(x > v.x) return true;
			if(y > v.y) return true;
			if(z > v.z) return true;
			return false;
		}
		// access
		float operator[](int i) const {
			ASSERT(i >= 0 && i <= 2);
			return (&x)[i];
		}
		float& operator[](int i) {
			ASSERT(i >= 0 && i <= 2);
			return (&x)[i];
		}
		Vector2 xy() {
			return Vector2(x, y);
		}
		// arithmetic
		Vector3 operator+(const Vector3& v) const {
			return Vector3(x + v.x, y + v.y, z + v.z);
		}
		Vector3& operator+=(const Vector3& v) {
			x += v.x;
			y += v.y;
			z += v.z;
			return *this;
		}
		Vector3 operator-(const Vector3& v) const {
			return Vector3(x - v.x, y - v.y, z - v.z);
		}
		Vector3& operator-=(const Vector3& v) {
			x -= v.x;
			y -= v.y;
			z -= v.z;
			return *this;
		}
		Vector3 operator*(float f) const {
			return Vector3(x * f, y * f, z * f);
		}
		Vector3& operator*=(float f) {
			x *= f;
			y *= f;
			z *= f;
			return *this;
		}
		Vector3 operator/(float f) const {
			CHECK_FLOAT_EQUAL(f, 0.f);
			float inv = 1.f / f;
			return Vector3(x * inv, y * inv, z * inv);
		}
		Vector3& operator/=(float f) {
			CHECK_FLOAT_EQUAL(f, 0.f);
			float inv = 1.f / f;
			x *= inv;
			y *= inv;
			z *= inv;
			return *this;
		}
		Vector3 operator -() const {
			return Vector3(-x, -y, -z);
		}
		// normalization
		float length2() const {
			return x * x + y * y + z * z;
		}
		float length() const {
			return sqrtf(length2());
		}
		bool HasNaNs() const;

		friend std::ostream& operator<<(std::ostream& os, const Vector3& v);
		float x, y, z;
};

inline Vector3 operator*(float f, const Vector3& v) {
	return v * f;
}

inline float dot(const Vector3& a, const Vector3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vector3 cross(const Vector3& a, const Vector3& b) {
	return Vector3((a.y * b.z) - (a.z * b.y),
			(a.z * b.x) - (a.x * b.z),
			(a.x * b.y) - (a.y * b.x));
}

inline Vector3 normalize(const Vector3& v) {
	return v / v.length();
}

class Vector4 {
	public:
		Vector4();
		explicit Vector4(float _x, float _y, float _z, float _w);
		// access
		float operator[](int i) const {
			ASSERT(i >= 0 && i <= 3);
			return (&x)[i];
		}
		float& operator[](int i) {
			ASSERT(i >= 0 && i <= 3);
			return (&x)[i];
		}
		Vector2 xy() {
			return Vector2(x, y);
		}
		Vector3 xyz() {
			return Vector3(x, y, z);
		}
		// arithmetic
		Vector4 operator+(const Vector4& v) const {
			return Vector4(x + v.x, y + v.y, z + v.z, w + v.w);
		}
		Vector4& operator+=(const Vector4& v) {
			x += v.x;
			y += v.y;
			z += v.z;
			w += v.w;
			return *this;
		}
		Vector4 operator-(const Vector4& v) const {
			return Vector4(x - v.x, y - v.y, z - v.z, w - v.w);
		}
		Vector4& operator-=(const Vector4& v) {
			x -= v.x;
			y -= v.y;
			z -= v.z;
			w -= v.w;
			return *this;
		}
		Vector4 operator*(float f) const {
			return Vector4(x * f, y * f, z * f, w * f);
		}
		Vector4& operator*=(float f) {
			x *= f;
			y *= f;
			z *= f;
			w *= f;
			return *this;
		}
		Vector4 operator/(float f) const {
			CHECK_FLOAT_EQUAL(f, 0.f);
			float inv = 1.f / f;
			return Vector4(x * inv, y * inv, z * inv, w * inv);
		}
		Vector4& operator/=(float f) {
			CHECK_FLOAT_EQUAL(f, 0.f);
			float inv = 1.f / f;
			x *= inv;
			y *= inv;
			z *= inv;
			w *= inv;
			return *this;
		}
		Vector4 operator -() const {
			return Vector4(-x, -y, -z, -w);
		}
		// normalization
		float length2() const {
			return x * x + y * y + z * z + w * w;;
		}
		float length() const {
			return sqrtf(length2());
		}
		bool HasNaNs() const;

		friend std::ostream& operator<<(std::ostream& os, const Vector4& v);
		float x, y, z, w;
};

typedef Vector2 vec2;
typedef Vector3 vec3;
typedef Vector4 vec4;

template<size_t D = 3, typename T = float>
	class Vector {
		public:
			Vector() {
				size = D;
				memset(v, 0, D * sizeof(T));
			}

			Vector(size_t n, const T *t)
				: Vector() {
					for(size_t i = 0; i < D && i < n; i++)
						v[i] = t[i];
				}

			Vector(const T& t)
				: Vector() {
					for(size_t i = 0; i < D; i++)
						v[i] = t;
				}

			Vector(const T& x, const T& y)
				: Vector() {
					if(size > 1) {
						v[0] = x;
						v[1] = y;
					}
				}

			Vector(const T& x, const T& y, const T& z)
				: Vector() {
					if(size > 2) {
						v[0] = x;
						v[1] = y;
						v[2] = z;
					}
				}

			T operator[](int i) const {
				ASSERT(i >= 0 && i <= static_cast<int>(size));
				return v[i];
			}
			T& operator[](int i) {
				ASSERT(i >= 0 && i <= static_cast<int>(size));
				return v[i];
			}
			bool operator==(const Vector<D, T>& _v) const {
				for(size_t i = 0; i < size; i++)
					if(!IS_EQUAL(v[i], _v[i]))
						return false;
				return true;
			}
			bool operator<=(const Vector<D, T>& _v) const {
				for(size_t i = 0; i < size; i++)
					if(v[i] > _v[i])
						return false;
				return true;
			}
			bool operator<(const Vector<D, T>& _v) const {
				for(size_t i = 0; i < size; i++)
					if(v[i] >= _v[i])
						return false;
				return true;
			}
			bool operator>=(const Vector<D, T>& _v) const {
				for(size_t i = 0; i < size; i++)
					if(v[i] < _v[i])
						return false;
				return true;
			}
			bool operator>(const Vector<D, T>& _v) const {
				for(size_t i = 0; i < size; i++)
					if(v[i] <= _v[i])
						return false;
				return true;
			}

			Vector<D, T> operator-(const Vector<D, T>& _v) const {
				Vector<D, T> v_;
				for(size_t i = 0; i < D; i++)
					v_[i] = v[i] - _v[i];
				return v_;
			}

			Vector<D, T> operator+(const Vector<D, T>& _v) const {
				Vector<D, T> v_;
				for(size_t i = 0; i < D; i++)
					v_[i] = v[i] + _v[i];
				return v_;
			}

			Vector<2, T> xy(size_t x = 0, size_t y = 1) const {
				return Vector<2, T>(v[x], v[y]);
			}

			Vector<2, T> floatXY(size_t x = 0, size_t y = 1) const {
				return Vector<2, T>(static_cast<float>(v[x]), static_cast<float>(v[y]));
			}

			Vector<3, T> floatXYZ(size_t x = 0, size_t y = 1, size_t z = 2) {
				return Vector<3, T>(static_cast<float>(v[x]),
														static_cast<float>(v[y]),
														static_cast<float>(v[z]));
			}

			T max() const {
				T m = v[0];
				for(size_t i = 1; i < D; i++)
					m = std::max(m, v[i]);
				return m;
			}

			friend std::ostream& operator<<(std::ostream& os, const Vector& v) {
				os << "Vector[<" << D << ">]";
				for(size_t i = 0; i < v.size; i++)
					os << v[i] << " ";
				os << std::endl;
				return os;
			}

			size_t size;
			T v[D];
	};

typedef Vector<2, int> ivec2;
typedef Vector<2, uint> uivec2;
typedef Vector<3, int> ivec3;
typedef Vector<3, uint> uivec3;
typedef Vector<4, int> ivec4;
typedef Vector<4, uint> uivec4;

	/* round
	 * @v **[in]** vector
	 * @return a vector with ceil applied to all components
	 */
	Vector<3, int> ceil(const Vector3 &v);
	/* round
	 * @v **[in]** vector
	 * @return a vector with floor applied to all components
	 */
	Vector<3, int> floor(const Vector3 &v);

	Vector<3, int> min(Vector<3, int> a, Vector<3, int> b);
	Vector<3, int> max(Vector<3, int> a, Vector<3, int> b);

} // ponos namespace

#endif
