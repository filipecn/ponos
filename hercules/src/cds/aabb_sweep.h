#ifndef HERCULES_CDS_AABB_SWEEP_H
#define HERCULES_CDS_AABB_SWEEP_H

#include <ponos.h>
#include <functional>

#include "cds/contact.h"
#include "cds/utils.h"

namespace hercules {

	namespace cds {

		/* method
		 * Implements _sweep and prune_ method to spatially sort objects.
		 */
		template<typename AABBObjectType>
			class AABBSweep : public ponos::CObjectPool<AABBObjectType> {
				public:
					AABBSweep()
					: ponos::CObjectPool<AABBObjectType>(), sortAxis(0) {}
					virtual ~AABBSweep() {}

					std::vector<Contact2D> collide() {
						if(list.size() < this->end)
							list.resize(this->end);
						for(size_t i = 0; i < this->end; i++)
							list[i] = i;
						std::vector<Contact2D> contacts;
						sortAndSweep(contacts);
						return contacts;
					}

				private:
					void sortAndSweep(std::vector<Contact2D>& contacts) {
						ponos::insertion_sort<size_t>(&list[0], this->end,
								[this](const size_t& a, const size_t& b) {
									ponos::BBox2D ba = this->pool[a].getWBBox();
									ponos::BBox2D bb = this->pool[b].getWBBox();
									if(IS_EQUAL(ba.pMin[sortAxis], bb.pMin[sortAxis]))
										return ba.pMax[sortAxis] < bb.pMax[sortAxis];
									return ba.pMin[sortAxis] < bb.pMin[sortAxis];
								});
						float s[3] = { 0.f, 0.f, 0.f }, s2[3] = { 0.f, 0.f, 0.f }, v[3];
						for(size_t i = 0; i < this->end; i++) {
							ponos::BBox2D curBBox = this->pool[list[i]].getWBBox();
							ponos::Point2 p = 0.5f * (curBBox.pMin + ponos::vec2(curBBox.pMax));
							for(int c = 0; c < 2; c++) {
								s[c] += p[c];
								s2[c] += p[c] * p[c];
							}
							for(size_t j = i + 1; j < this->end; j++) {
								ponos::BBox2D cBBox = this->pool[list[j]].getWBBox();
								if(cBBox.pMin[sortAxis] > curBBox.pMax[sortAxis])
									break;
                if (ponos::bbox_bbox_intersection(curBBox, cBBox)) {
                  Contact2D contact = _testCollision(list[i], list[j]);
                  if (contact.valid) {
                    contact.a = &this->pool[list[i]];
                    contact.b = &this->pool[list[j]];
                    contacts.emplace_back(contact);
                  }
								}
							}
						}
						for(int c = 0; c < 2; c++)
							v[c] = s2[c] - s[c] * s[c] / this->end;
						sortAxis = 0;
						if(v[1] > v[0])
							sortAxis = 1;
					}

					Contact2D _testCollision(size_t a, size_t b) {
						return compute_collision(this->pool[a].getShape(),
                                     this->pool[b].getShape());
					}

					size_t sortAxis;
					std::vector<size_t> list;
			};

	} // cds namespace

} // hercules namespace

#endif // HERCULES_CDS_AABB_SWEEP_H
