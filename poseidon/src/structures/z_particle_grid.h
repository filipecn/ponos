#ifndef POSEIDON_STRUCTURES_Z_PARTICLE_GRID_H
#define POSEIDON_STRUCTURES_Z_PARTICLE_GRID_H

#include "elements/particle.h"

#include <ponos.h>

#include <algorithm>
#include <functional>
#include <vector>

namespace poseidon {

/** Keep particles sorted in Z order for fast neighbour search operations.
 */
template <typename ParticleObject = FLIPParticle2D> class ZParticleGrid2D {
public:
  ZParticleGrid2D() {
    end = 0;
    tree_ = nullptr;
  }
  /** \brief Constructor
   * \param w **[in]** width (**power of 2**)
   * \param h **[in]** height (**power of 2**)
   * \param t **[in]** scale and offset
   */
  ZParticleGrid2D(size_t w, size_t h, const ponos::Transform2D &t)
      : ZParticleGrid2D() {
    set(w, h, t);
  }
  /** \brief Constructor
   * \param w **[in]** width (**power of 2**)
   * \param h **[in]** height (**power of 2**)
   * \param bbox **[in]** region in space
   */
  ZParticleGrid2D(size_t w, size_t h, const ponos::BBox2D &bbox)
      : ZParticleGrid2D() {
    ponos::Transform2D t =
        ponos::translate(ponos::vec2(bbox.pMin)) *
        ponos::scale((bbox.pMax[0] - bbox.pMin[0]) / static_cast<float>(w),
                     (bbox.pMax[1] - bbox.pMin[1]) / static_cast<float>(h));
    set(w, h, t);
  }
  virtual ~ZParticleGrid2D() {}

  struct ParticleElement {
    // template <typename... Args> ParticleElement(Args &&... args) {
    //  new (&data) ParticleObject(std::forward<Args>(args)...);
    //  active = true;
    //}
    ParticleElement() { active = true; }
    void setPosition(ZParticleGrid2D grid, const ponos::Point2 &p) {
      data.position = p;
      ponos::Point2 gp = grid.toGrid(p);
      zcode = ponos::mortonCode(gp.x, gp.y);
    }
    ParticleObject data;
    uint32 zcode;
    bool active;
  };

  class particle_iterator {
  public:
    particle_iterator(ZParticleGrid2D<ParticleObject> &g, size_t f = 0,
                      size_t depth = 0)
        : first(f), cur(0), last(0), grid(g) {
      comp = [](const ParticleElement &p, const uint32 &v) {
        if (p.zcode < v)
          return -1;
        if (p.zcode > v)
          return 1;
        return 0;
      };
      if (grid.end == 0)
        return;

      int fi = ponos::lower_bound<ParticleElement, uint32>(
          &grid.particles[0], grid.end, first, comp);
      last = grid.end;
      if (depth) {
        last = first + (1 << ((grid.nbits - depth) * 2));
        last = ponos::lower_bound<ParticleElement, uint32>(
                   &grid.particles[0], grid.end, last, comp) +
               1;
      }
      first = fi + 1;
      cur = first;
    }
    bool next() const { return cur < last; }
    ParticleObject *get() {
      if (cur >= last)
        return nullptr;
      return &grid.particles[cur].data;
    }
    ParticleObject *operator*() {
      if (cur >= last)
        return nullptr;
      return &grid.particles[cur].data;
    }
    ParticleElement *particleElement() { return &grid.particles[cur]; }

    void operator++() { cur++; }
    int count() const { return last - first; }

  private:
    size_t first, cur, last;
    ZParticleGrid2D<ParticleObject> &grid;
    std::function<int(const ParticleElement &p, const uint32 &v)> comp;
  };

  class tree {
  public:
    tree(ZParticleGrid2D<ParticleObject> &g,
         std::function<bool(uint32 id, uint32 depth)> f)
        : grid(g) {
      //	grid.update();
      root = new Node(ponos::BBox2D(ponos::Point2(),
                                    ponos::Point2(grid.width, grid.height)),
                      0, 0);
      root->first = 0;
      refine(f);
    }
    ~tree() { destroy(root); }
    void refine(std::function<bool(uint32 id, uint32 depth)> f) {
      refine_(f, root, 1);
    }
    void iterateParticles(const ponos::BBox2D &bbox,
                          std::function<void(ParticleObject *o)> f) {
      bbox_search(root, bbox, grid.toGrid(bbox), f);
    }
    void update() {}

    template <typename FilterType>
    void searchAndApply(const ponos::Point2 &p, float radius, FilterType &f) {
      const ponos::BBox2D bbox(p - ponos::vec2(radius),
                               p + ponos::vec2(radius));
      iterateParticles(bbox, [&p, &f](ParticleObject *o) {
        float sd = ponos::distance2(o->position, p);
        f(sd, o);
      });
    }

    struct Node {
      Node(ponos::BBox2D b, int d, uint32 _id) : bbox(b), depth(d), id(_id) {
        for (int i = 0; i < 4; i++)
          child[i] = nullptr;
        first = last = -1;
      }
      ponos::BBox2D bbox;
      int depth;
      uint32 id;
      int first, last;
      Node *child[4];
    };

    Node *root;

  private:
    void destroy(Node *node) {
      if (!node)
        return;
      for (int i = 0; i < 4; i++)
        destroy(node->child[i]);
      delete node;
    }
    void refine_(std::function<bool(uint32 id, uint32 depth)> f, Node *node,
                 uint32 depth) {
      if (depth > grid.maxDepth)
        return;
      if (!node)
        return;
      if (!f(node->id, node->depth))
        return;
      ponos::Point2 center = node->bbox.center();
      float x[3] = {node->bbox.pMin.x, center.x, node->bbox.pMax.x};
      float y[3] = {node->bbox.pMin.y, center.y, node->bbox.pMax.y};
      /*std::cout << "breaking id with depth " << depth << "\n";
         ponos::printBits(node->id);        std::cout << std::endl;
              ponos::printBits(node->id | (0 << ((grid.nbits - depth) * 2)));
         std::cout << std::endl;
              ponos::printBits(node->id | (1 << ((grid.nbits - depth) * 2)));
         std::cout << std::endl;
              ponos::printBits(node->id | (2 << ((grid.nbits - depth) * 2)));
         std::cout << std::endl;
              ponos::printBits(node->id | (3 << ((grid.nbits - depth) * 2)));
         std::cout << std::endl;
              std::cout << std::endl;
              std::cout << std::endl;*/
      node->child[0] = new Node(
          ponos::BBox2D(ponos::Point2(x[0], y[0]), ponos::Point2(x[1], y[1])),
          depth, node->id | (0 << ((grid.nbits - depth) * 2)));
      node->child[1] = new Node(
          ponos::BBox2D(ponos::Point2(x[1], y[0]), ponos::Point2(x[2], y[1])),
          depth, node->id | (1 << ((grid.nbits - depth) * 2)));
      node->child[2] = new Node(
          ponos::BBox2D(ponos::Point2(x[0], y[1]), ponos::Point2(x[1], y[2])),
          depth, node->id | (2 << ((grid.nbits - depth) * 2)));
      node->child[3] = new Node(
          ponos::BBox2D(ponos::Point2(x[1], y[1]), ponos::Point2(x[2], y[2])),
          depth, node->id | (3 << ((grid.nbits - depth) * 2)));
      for (int i = 0; i < 4; i++)
        refine_(f, node->child[i], depth + 1);
    }
    void bbox_search(Node *node, const ponos::BBox2D &bbox,
                     const ponos::BBox2D &gbbox,
                     std::function<void(ParticleObject *o)> f) {
      if (!node)
        return;
      if (ponos::bbox_bbox_intersection(gbbox, node->bbox)) {
        if (!node->child[0] || !node->child[1] || !node->child[2] ||
            !node->child[3]) {
          particle_iterator it(grid, node->id, node->depth);
          while (it.next()) {
            if (bbox.inside((*it)->position))
              f(*it);
            ++it;
          }
        } else {
          for (int i = 0; i < 4; i++)
            bbox_search(node->child[i], bbox, gbbox, f);
        }
      }
    }
    ZParticleGrid2D<ParticleObject> &grid;
  };

  template <typename T> struct WeightedAverageAccumulator {
    typedef T ValueType;
    WeightedAverageAccumulator(const T radius, ParticleAttribute a)
        : ra(radius), invRadius(1.0 / radius), weightSum(0.0), valueSum(0.0),
          attribute(a) {}

    void reset() { weightSum = valueSum = T(0.0); }

    void operator()(const T distSqr, ParticleObject *p) {
      if (p->type != ParticleTypes::FLUID)
        return;
      // std::cout << p->position << p->velocity << std::endl;
      // std::cout << distSqr << " " << ra << std::endl;
      float w = p->mass * ponos::sharpen(distSqr, ra);
      w = exp(-distSqr / (2.0 * SQR(ra / 4.0)));
      float value = 0.f;
      switch (attribute) {
      case ParticleAttribute::VELOCITY_X:
        value = p->velocity.x;
        break;
      case ParticleAttribute::VELOCITY_Y:
        value = p->velocity.y;
        break;
      case ParticleAttribute::DENSITY:
        value = p->density;
        break;
      default:
        break;
      }
      weightSum += w;
      valueSum += w * value;
    }

    T result() const {
      return weightSum > T(0.0) ? valueSum / weightSum : T(0.0);
    }

  private:
    const T ra;
    const T invRadius;
    T weightSum, valueSum;
    ParticleAttribute attribute;
  };

  /** \brief set
   * \param w **[in]** width (**power of 2**)
   * \param h **[in]** height (**power of 2**)
   * \param t **[in]** scale and offset
   */
  void set(size_t w, size_t h, const ponos::Transform2D &t) {
    width = w;
    height = h;
    dimensions = ponos::ivec2(w, h);
    toWorld = t;
    toGrid = ponos::inverse(t);
    int n = std::max(width, height) - 1;
    for (nbits = 0; n; n >>= 1) {
      nbits++;
    }
    maxDepth = nbits;
  }
  /** add
   *
   * Particle positions are given in world coordinates
   */
  template <typename... Args> ParticleObject *add(Args &&... args) {
    if (end == particles.size())
      // particles.emplace_back(std::forward<Args>(args)...);
      particles.emplace_back();
    // else
    particles[end].active = true;
    new (&particles[end].data) ParticleObject(std::forward<Args>(args)...);
    particles[end].zcode = computeIndex(toGrid(particles[end].data.position));
    end++;
    return &particles[end - 1].data;
  }

  void addParticle(ParticleObject *p) {
    if (end == particles.size())
      particles.emplace_back();
    particles[end].active = true;
    particles[end].data = *p;
    particles[end].zcode = computeIndex(toGrid(particles[end].data.position));
    end++;
  }

  void update() {
    if (!particles.size())
      return;
    ponos::parallel_for(0, end, [this](size_t s, size_t e) {
      for (size_t i = s; i <= e; i++)
        this->particles[i].zcode =
            computeIndex(toGrid(this->particles[i].data.position));
    });
    std::sort(&particles[0], &particles[0] + end,
              [](const ParticleElement &a, const ParticleElement &b) {
                if (!a.active)
                  return false;
                if (!b.active)
                  return true;
                if (a.zcode > b.zcode)
                  return false;
                if (a.zcode == b.zcode) {
                  if (a.data.position.x > b.data.position.x)
                    return false;
                  if (a.data.position.x < b.data.position.x)
                    return true;
                  return a.data.position.y < b.data.position.y;
                }
                return true;
              });
    // compute new end in case some particles have been deleted
    if (end > 0) {
      while (!particles[end - 1].active)
        end--;
    }
    if (tree_)
      delete tree_;
    tree_ = new tree(*this, [](uint32 id, uint32 depth) { return true; });
  }

  // queries
  particle_iterator getCell(const ponos::ivec2 &ij) {
    particle_iterator it(*this, computeIndex(ponos::Point2(ij[0], ij[1])),
                         maxDepth);
    return it;
  }

  void iterateAll(std::function<void(ParticleObject *p)> f) {
    // ponos::parallel_for(0, end, [this, f](size_t s, size_t e) {
    //  for (size_t i = s; i <= e; i++)
    //    f(&this->particles[i].data);
    //});
    for (size_t i = 0; i < end; i++)
      f(&this->particles[i].data);
  }

  void dumpToFile(const char *filename) {
    FILE *fp = fopen(filename, "w+");
    for (size_t i = 0; i < end; i++) {
      if (particles[i].data.type != ParticleTypes::FLUID)
        continue;
      fprintf(fp, "%f %f\n", particles[i].data.position.x,
              particles[i].data.position.y);
    }
    fclose(fp);
  }
  /* \brief gather
   * \param a **[in]** attribute
   * \param p **[in]** center (world coordinates)
   * \param r **[in]** radius (ex: value is **1.5 * cellSize**
   * Computes weighted average value of particles atttribute values within
   * **r** distance from **p**.
   * \return gathered value
   */
  float gather(ParticleAttribute a, const ponos::Point2 &p, float r) const {
    WeightedAverageAccumulator<float> accumulator(r, a);
    tree_->searchAndApply(p, r, accumulator);
    return accumulator.result();
  }

  size_t elementCount() { return particles.size(); }

  ponos::ivec2 dimensions;
  size_t width, height;
  ponos::Transform2D toGrid, toWorld;
  tree *tree_;

private:
  size_t end;
  std::vector<ParticleElement> particles;

  uint32 nbits, maxDepth;

  // position in index space
  // position in grid coordinates
  static uint32 computeIndex(const ponos::Point2 &p) {
    return ponos::mortonCode(p.x, p.y);
  }
};

} // poseidon namespace

#endif // POSEIDON_STRUCTURES_Z_PARTICLE_GRID_H
