#ifndef HERCULES_CDS_GJK_H
#define HERCULES_CDS_GJK_H

#include <ponos.h>

namespace hercules {

	namespace cds {

		class GJK {
			public:
			static bool intersect(const ponos::Polygon& a, const ponos::Polygon& b) {
				ponos::vec2 D(1, 0);
				ponos::vec2 s[3];
				int k = 0;
				int z = 0;
				for(;;) {
					ponos::vec2 p = support(a, D) - support(b, -D);
					if(ponos::dot(ponos::vec2(p), D) < 0)
						return false;
					if(buildSimplex(s, k, p, D))
						return true;
					z++;
					if(z > 100)
						break;
				}
				return false;
			}
			private:
			GJK() {}
			static ponos::vec2 support(const ponos::Polygon& a, const ponos::vec2& D) {
				size_t size = a.vertices.size();
				float M = ponos::dot(ponos::vec2(a.vertices[0]), D);
				size_t Mi = 0;
				for(size_t i = 1; i < size; i++) {
					float cM = ponos::dot(ponos::vec2(a.vertices[i]), D);
					if(cM > M) {
						M = cM;
						Mi = i;
					}
				}
				return ponos::vec2(a.vertices[Mi]);
			}
			static bool buildSimplex(ponos::vec2 s[], int &k, ponos::vec2 p, ponos::vec2& D) {
				k = std::min(k + 1, 3);
				s[k - 1] = p;
				if(k == 1) {
					D = -s[0];
					return false;
				}
				if(k == 2) {
					ponos::vec2 a = s[1] - s[0];
					if(cross(a,-s[0]) < 0) {
						s[2] = s[1];
						s[1] = s[0];
						s[0] = s[2];
						D = a.right();
					}
					else D = a.left();
					return false;
				}
				ponos::vec2 a = s[2] - s[0];
				if(cross(a, -s[0]) > 0) {
					D = a.left();
					s[1] = s[2];
					k--;
					return false;
				}
				ponos::vec2 b = s[2] - s[1];
				if(cross(b, -s[1]) < 0) {
					D = b.right();
					s[0] = s[2];
					k--;
					return false;
				}
				return true;
			}
		};

	} // cds namespace

} // hercules namespace

#endif // HERCULES_CDS_GJK_H

