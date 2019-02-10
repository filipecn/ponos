// Created by filipecn on 2019-01-06.
/*
 * Copyright (c) 2019 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#include "animated_transform.h"

namespace helios {

AnimatedTransform::AnimatedTransform(const HTransform *t1, real_t time1,
                                     const HTransform *t2, real_t time2)
    : startTime(time1), endTime(time2), startTransform(t1), endTransform(t2),
      actuallyAnimated(*startTransform != *endTransform) {
  decompose(startTransform->matrix(), &T[0], &R[0], &S[0]);
  decompose(endTransform->matrix(), &T[1], &R[1], &S[1]);
  // flip R[1] if needed to select shortest path
  if (ponos::dot(R[0], R[1]) < 0)
    R[1] = -R[1];
  hasRotation = ponos::dot(R[0], R[1]) < 0.9995f;
  // compute terms of motion derivative functions
}

AnimatedTransform::~AnimatedTransform() {}

void AnimatedTransform::decompose(const ponos::mat4 &m, ponos::vec3 *T,
                                  ponos::Quaternion *Rquat, ponos::mat4 *s) {
  // extract translation T from transformation matrix
  T->x = m.m[0][3];
  T->y = m.m[1][3];
  T->z = m.m[2][3];
  // compute new transformation matrix M without translation
  ponos::mat4 M = m;
  for (int i = 0; i < 3; i++)
    M.m[i][3] = M.m[3][i] = 0.f;
  M.m[3][3] = 1.f;
  ponos::mat4 r;
  // extract rotation R from transformation matrix
  // and scale S
  ponos::decompose(M, r, *s);
  *Rquat = ponos::Quaternion(r);
}

} // namespace helios