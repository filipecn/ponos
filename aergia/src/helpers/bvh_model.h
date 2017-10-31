#ifndef AERGIA_HELPERS_BVH_MODEL_H
#define AERGIA_HELPERS_BVH_MODEL_H

#include <ponos.h>

#include "helpers/geometry_drawers.h"
#include "scene/bvh.h"
#include "scene/scene_object.h"
#include "utils/open_gl.h"

namespace aergia {

/* bvh model
 * Draw BVH nodes
 */
class BVHModel : public SceneObject {
public:
  BVHModel(const BVH *_bvh) : bvh(_bvh) {}
  /* @inherit */
  void draw() const override {
    ponos::Transform inv = ponos::inverse(bvh->sceneMesh->transform);
    ponos::Ray3 r(ponos::Point3(0, 0, 0), ponos::vec3(1, 1, 1));
    glBegin(GL_LINES);
    aergia::glVertex(r.o);
    aergia::glVertex(r.o + 1000.f * r.d);
    glEnd();
    recDraw(inv(r), bvh->root);
    return;
    static int k = 0;
    static int t = 0;
    t++;
    if (t > 1000) {
      t = 0;
      k = (k + 1) % bvh->nodes.size();
    }
    glLineWidth(1.f);
    for (size_t i = 0; i < bvh->nodes.size(); i++) {
      // if(i != k) continue;
      glColor4f(0, 0, 1, 0.4);
      if (bvh->nodes[i].nElements == 1)
        glColor4f(1, 0, 0, 0.8);
      draw_bbox(bvh->sceneMesh->transform(bvh->nodes[i].bounds));
    }
  }

  void recDraw(ponos::Ray3 r, BVH::BVHNode *n) const {
    if (!n)
      return;
    float a, b;
    if (ponos::bbox_ray_intersection(n->bounds, r, a, b)) {
      // if(!(n->children[0] || n->children[1])) {
      glColor4f(0, 0, 1, 1);
      draw_bbox(bvh->sceneMesh->transform(n->bounds));
      glColor4f(0, 0, 0, 0.3);
      //}
      // else{
      recDraw(r, n->children[0]);
      recDraw(r, n->children[1]);
      //}
    }
  }

  bool intersect(const ponos::Ray3 &r, float *t = nullptr) override {
    UNUSED_VARIABLE(t);
    ray = r;
    return false;
  }

private:
  const BVH *bvh;
  ponos::Ray3 ray;
};

} // aergia namespace

#endif // AERGIA_HELPERS_CARTESIAN_GRID_H
