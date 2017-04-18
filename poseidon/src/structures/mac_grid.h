#ifndef POSEIDON_STRUCTURES_MAC_GRID_H
#define POSEIDON_STRUCTURES_MAC_GRID_H

#include <ponos.h>

#include <memory.h>

namespace poseidon {

enum class CellType { FLUID, AIR, SOLID };

/** \brief Mac-Grid structure.
 * Stores a staggered grid with velocity components on faces and the rest of
 * quantities at centers.
 */
template <class GridType = ponos::RegularGrid<float>> class MacGrid {
public:
  virtual ~MacGrid() {}
  /* setter
   * @d **[in]** dimensions
   * @s **[in]** scale
   * @o **[in]** offset
   *
   * Set macgrid dimensions and transform.
   */
  void set(const ponos::ivec3 &d, float s, ponos::vec3 o);

  // dimensions
  ponos::ivec3 dimensions;
  // velocity's x component
  std::shared_ptr<GridType> v_x;
  // velocity's y component
  std::shared_ptr<GridType> v_y;
  // velocity's z component
  std::shared_ptr<GridType> v_z;
  // pressure
  std::shared_ptr<GridType> p;
  // divergence
  std::shared_ptr<GridType> d;
  // cell type
  std::shared_ptr<ponos::RegularGrid<CellType>> cellType;
};

template <template <typename T> class GridType> class MacGrid2D {
public:
  MacGrid2D(size_t w, size_t h, float s, ponos::vec2 = ponos::vec2()) {
    set(ponos::ivec2(w, h),
        ponos::BBox2D(ponos::Point2(0, 0), ponos::Point2(w * s, h * s)));
  }
  MacGrid2D(size_t w, size_t h, const ponos::BBox2D &bbox) {
    set(ponos::ivec2(w, h), bbox);
  }

  virtual ~MacGrid2D() {}
  /** \brief setter
   * \param d **[in]** dimensions
   * \param s **[in]** scale
   * \param o **[in]** offset
   *
   * Set macgrid dimensions and transform.
   */
  void set(const ponos::ivec2 &d, float s, ponos::vec2 o);
  void set(const ponos::ivec2 &d, const ponos::BBox2D &bbox) {
    dx = bbox.size(0) / d[0];
    ponos::vec2 offset =
        0.5f *
        ponos::vec2((bbox.pMax[0] - bbox.pMin[0]) / static_cast<float>(d[0]),
                    (bbox.pMax[1] - bbox.pMin[1]) / static_cast<float>(d[1]));
    ponos::Transform2D t =
        ponos::translate(ponos::vec2(bbox.pMin + offset)) *
        ponos::scale((bbox.pMax[0] - bbox.pMin[0]) / static_cast<float>(d[0]),
                     (bbox.pMax[1] - bbox.pMin[1]) / static_cast<float>(d[1]));
    dimensions = d;
    toWorld = t;
    toGrid = ponos::inverse(t);
    v_u.reset(new GridType<float>(dimensions[0] + 1, dimensions[1]));
    v_u->setTransform(toWorld * ponos::translate(ponos::vec2(-0.5f, 0.0f)));
    v_v.reset(new GridType<float>(dimensions[0], dimensions[1] + 1));
    v_v->setTransform(toWorld * ponos::translate(ponos::vec2(0.0f, -0.5f)));
    p.reset(new GridType<float>(dimensions[0], dimensions[1]));
    p->setTransform(toWorld);
    D.reset(new GridType<float>(dimensions[0], dimensions[1]));
    D->setTransform(toWorld);
    cellType.reset(new GridType<CellType>(dimensions[0], dimensions[1]));
    cellType->setTransform(toWorld);
    cellType->border = CellType::SOLID;
  }

  void computeNegativeDivergence() {
    ponos::ivec2 ij;
    float s = 1.f / dx;
    D->setAll(0.f);
    FOR_INDICES0_2D(dimensions, ij) {
      if ((*cellType)(ij) == CellType::FLUID)
        (*D)(ij) = -s * ((*v_u)(ij[0] + 1, ij[1]) - (*v_u)(ij) +
                         (*v_v)(ij[0], ij[1] + 1) - (*v_v)(ij));
    }
  }

  ponos::vec2 sampleVelocity(const ponos::Point2 &wp) {
    return ponos::vec2(v_u->sample(wp.x, wp.y), v_v->sample(wp.x, wp.y));
  }

  ponos::Transform2D toWorld, toGrid;
  // cell size
  float dx;
  // dimensions
  ponos::ivec2 dimensions;
  // velocity's x component
  std::shared_ptr<GridType<float>> v_u;
  // velocity's y component
  std::shared_ptr<GridType<float>> v_v;
  // pressure
  std::shared_ptr<GridType<float>> p;
  // divergence
  std::shared_ptr<GridType<float>> D;
  // cell type
  std::shared_ptr<GridType<CellType>> cellType;
};

} // poseidon namespace

#endif // POSEIDON_STRUCTURES_MAC_GRID_H
