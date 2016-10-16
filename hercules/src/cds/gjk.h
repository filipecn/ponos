#ifndef HERCULES_CDS_GJK_H
#define HERCULES_CDS_GJK_H

#include <ponos.h>

namespace hercules {

	namespace cds {

		class GJK {
			bool intersect(const ponos::Polygon& a, const ponos::Polygon& b) {
				ponos::vec2 D(1, 0);
				std::vector<ponos::vec2> s;
				for(;;) {
					ponos::vec2 p = ponos::vec2(support(a, b, D));
					if(ponos::dot(p, D) < 0)
						return false;
					s.emplace_back(p);
					if(buildSimplex(s, D))
						return true;
				}
				return false;
			}
			private:
			ponos::Point2 support(const ponos::Polygon& a, const ponos::Polygon& b, const ponos::vec2& D) {
				return support(a, D) - ponos::vec2(support(b, -D));
			}
			ponos::Point2 support(const ponos::Polygon& a, const ponos::vec2& D) {
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
				return a.vertices[Mi];
			}
			bool buildSimplex(std::vector<ponos::vec2>& s, ponos::vec2& D) {
				if(s.size() == 1) {
					D = -ponos::vec2(s[0]);
					return false;
				}
				if(s.size() == 2) {
					ponos::vec2 a = s[1];
					ponos::vec2 b = s[0];
					D = cross_aba(b - a, -a);
					return false;
				}
				if(s.size() == 3) {
					ponos::vec2 a = s[2];
					ponos::vec2 b = s[1];
					ponos::vec2 c = s[0];
					ponos::vec2 ao = -a;
					ponos::vec2 ab = b - a;
					ponos::vec2 ac = c - a;
					ponos::vec2 abc = ponos::cross(ab, ac);
					ponos::vec2 abp = ponos::cross(ab, abc);
					if(ponos::dot(abp, ao) > 0) {
						D = cross_aba(ab, ao);
						return false;
					}
					ponos::vec2 acp = ponos::cross(abc, ac);
					if(ponos::dot(acp, ao) > 0) {
						D = cross_aba(ac, ao);
						return false;
					}
					return true;
				}
			}
			ponos::vec2 cross_aba(const ponos::vec2 &a, const ponos::vec2 &b) {
				return ponos::cross(ponos::cross(a, b), a);
			}
		};

	} // cds namespace

} // hercules namespace

#endif // HERCULES_CDS_GJK_H

