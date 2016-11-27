#ifndef POSEIDON_STRUCTURES_Z_PARTICLE_GRID_H
#define POSEIDON_STRUCTURES_Z_PARTICLE_GRID_H

#include "elements/particle.h"

#include <ponos.h>

#include <algorithm>
#include <functional>
#include <vector>

namespace poseidon {

	/* structure
	 * Keep particles sorted in Z order for fast neighbour search operations.
	 */
	template<typename ParticleObject = FLIPParticle2D>
		class ZParticleGrid2D {
			public:
				ZParticleGrid2D() {
					end = 0;
				}
				/* Constructor
				 * @w **[in]** width (**power of 2**)
				 * @h **[in]** height (**power of 2**)
				 * @t **[in]** scale and offset
				 */
				ZParticleGrid2D(size_t w, size_t h, const ponos::Transform2D& t)
					: ZParticleGrid2D() {
						set(w, h, t);
					}
				virtual ~ZParticleGrid2D() {}

				struct ParticleElement {
				template<typename... Args>
					ParticleElement(Args&&... args) {
							new (&data) ParticleObject(std::forward<Args>(args)...);
							zcode = computeIndex(data.position);
							active = true;
					}
					ParticleObject data;
					uint32 zcode;
					bool active;
				};

				class particle_iterator {
					public:
						particle_iterator(ZParticleGrid2D<ParticleObject>& g, size_t f = 0, size_t depth = 0)
							: first(f), cur(0), grid(g) {
								comp = [](const ParticleElement& p, const uint32& v) {
												if(p.zcode < v) return -1;
												if(p.zcode > v) return 1;
												return 0;
											};
								int fi = ponos::lower_bound<ParticleElement, uint32>(&grid.particles[0], grid.end, first, comp);
								last = grid.end;
								if(depth) {
									last = first + (1 << ((grid.nbits - depth) * 2));
									last = ponos::lower_bound<ParticleElement, uint32>(&grid.particles[0], grid.end, last, comp) + 1;
								}
								first = fi + 1;
							}
						bool next() {
							return cur < last;
						}
						ParticleObject* get() {
							if(cur >= last)
								return nullptr;
							return &grid.particles[cur].data;
						}
						ParticleObject* operator*() {
              if(cur >= last)
								return nullptr;
              return &grid.particles[cur].data;
						}
						void operator++() {
							cur++;
						}
						int count() {
							return last - first;
						}

					private:
						size_t first, last;
						size_t cur;
						ZParticleGrid2D<ParticleObject>& grid;
						std::function<int(const ParticleElement& p, const uint32& v)> comp;
				};

				class tree {
					public:
						tree(ZParticleGrid2D<ParticleObject>& g, std::function<bool(uint32 id, uint32 depth)> f)
							: grid(g) {
								grid.update();
								root = new Node(ponos::BBox2D(ponos::Point2(-.5f, -.5f), ponos::Point2(grid.width, grid.height) + ponos::vec2(.5f, .5f)), 0, 0);
								root->first = 0;
								refine(f);
							}
						~tree() {
							destroy(root);
						}
						void refine(std::function<bool(uint32 id, uint32 depth)> f) {
							refine_(f, root, 1);
						}
            void iterateParticles(const ponos::BBox2D& bbox, std::function<void(ParticleObject* o)> f) {
							bbox_search(root, bbox, f);
            }
						void update() {}

						struct Node {
							Node(ponos::BBox2D b, int d, uint32 _id)
							: bbox(b), depth(d), id(_id) {
									for(int i = 0; i < 4; i++)
										child[i] = nullptr;
									first = last = -1;
								}
							ponos::BBox2D bbox;
							int depth;
							uint32 id;
							int first, last;
							Node* child[4];
						};

						private:
						void destroy(Node* node) {
							if(!node)
								return;
							for(int i = 0; i < 4; i++)
								destroy(node->child[i]);
							delete node;
						}
						void refine_(std::function<bool(uint32 id, uint32 depth)> f, Node* node, uint32 depth) {
							if(depth > grid.maxDepth)
								return;
							if(!node)
								return;
							if(!f(node->id, node->depth))
								return;
							ponos::Point2 center = node->bbox.center();
							float x[3] = {node->bbox.pMin.x, center.x, node->bbox.pMax.x};
							float y[3] = {node->bbox.pMin.y, center.y, node->bbox.pMax.y};
							/*std::cout << "breaking id with depth " << depth << "\n"; ponos::printBits(node->id);        std::cout << std::endl;
							ponos::printBits(node->id | (0 << ((grid.nbits - depth) * 2))); std::cout << std::endl;
							ponos::printBits(node->id | (1 << ((grid.nbits - depth) * 2))); std::cout << std::endl;
							ponos::printBits(node->id | (2 << ((grid.nbits - depth) * 2))); std::cout << std::endl;
							ponos::printBits(node->id | (3 << ((grid.nbits - depth) * 2))); std::cout << std::endl;
							std::cout << std::endl;
							std::cout << std::endl;*/
							node->child[0] = new Node(ponos::BBox2D(ponos::Point2(x[0], y[0]), ponos::Point2(x[1], y[1])), depth, node->id | (0 << ((grid.nbits - depth) * 2)));
							node->child[1] = new Node(ponos::BBox2D(ponos::Point2(x[1], y[0]), ponos::Point2(x[2], y[1])), depth, node->id | (1 << ((grid.nbits - depth) * 2)));
							node->child[2] = new Node(ponos::BBox2D(ponos::Point2(x[0], y[1]), ponos::Point2(x[1], y[2])), depth, node->id | (2 << ((grid.nbits - depth) * 2)));
							node->child[3] = new Node(ponos::BBox2D(ponos::Point2(x[1], y[1]), ponos::Point2(x[2], y[2])), depth, node->id | (3 << ((grid.nbits - depth) * 2)));
							for(int i = 0; i < 4; i++)
								refine_(f, node->child[i], depth + 1);
						}
            void bbox_search(Node* node, const ponos::BBox2D& bbox, std::function<void(ParticleObject* o)> f) {
              if (!node)
                return;
              if (ponos::bbox_bbox_intersection(bbox, node->bbox)) {
                if (!node->child[0] || !node->child[1] || !node->child[2] || !node->child[3]) {
                  particle_iterator it(grid, node->id, node->depth);
                  while (it.next()) {
                    f(*it);
                    ++it;
                  }
                }
                else {
                  for (int i = 0; i < 4; i++)
                    bbox_search(node->child[i], bbox, f);
                }
              }
            }
						ZParticleGrid2D<ParticleObject>& grid;
						Node* root;
				};

				/* set
				 * @w **[in]** width (**power of 2**)
				 * @h **[in]** height (**power of 2**)
				 * @t **[in]** scale and offset
				 */
				void set(size_t w, size_t h, const ponos::Transform2D& t) {
					width = w;
					height = h;
					transform = t;
					int n = std::max(width, height) - 1;
					for(nbits = 0; n; n >>= 1) { nbits++; }
					maxDepth = nbits;
				}
				/* add
				 * Particle positions are given in index coordinates
				 * (**[0, width]x[0, height]**)
				 */
				template<typename... Args>
					void add(Args&&... args) {
						if (end == particles.size())
							particles.emplace_back(std::forward<Args>(args)...);
						else
							new (&particles[end].data) ParticleObject(std::forward<Args>(args)...);
						end++;
					}

				void update() {
					std::sort(&particles[0], &particles[0] + end, [](const ParticleElement& a, const ParticleElement& b){
							if(a.zcode > b.zcode)
								return false;
							if(a.zcode == b.zcode) {
								if(a.data.position.x > b.data.position.x)
									return false;
								if(a.data.position.x < b.data.position.x)
									return true;
								return a.data.position.y < b.data.position.y;
							}
							return true;
					});
				}

			private:
				size_t end;
				std::vector<ParticleElement> particles;
				size_t width, height;
				ponos::Transform2D transform;
				uint32 nbits, maxDepth;

				static uint32 separate_by_1(uint32 n) {
					n = (n ^ (n << 8)) & 0x00ff00ff;
					n = (n ^ (n << 4)) & 0x0f0f0f0f;
					n = (n ^ (n << 2)) & 0x33333333;
					n = (n ^ (n << 1)) & 0x55555555;
					return n;
				}

				static uint32 morton_code(uint32 x, uint32 y) {
					return (separate_by_1(y) << 1) + separate_by_1(x);
				}
				// position in index space
				static uint32 computeIndex(const ponos::Point2& pos) {
					return morton_code(static_cast<int>(pos.x + 0.5f), static_cast<int>(pos.y + 0.5f));
				}
		};

} // poseidon namespace

#endif // POSEIDON_STRUCTURES_Z_PARTICLE_GRID_H
