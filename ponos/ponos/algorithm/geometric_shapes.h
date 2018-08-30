// Created by filipecn on 3/14/18.
/*
 * Copyright (c) 2018 FilipeCN
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

#ifndef PONOS_GEOMETRIC_SHAPES_H
#define PONOS_GEOMETRIC_SHAPES_H

#include <ponos/structures/raw_mesh.h>

namespace ponos {

///
/// \param center
/// \param radius
/// \param divisions
/// \param generateNormals
/// \param generateUVs
/// \return raw mesh pointer
RawMesh *create_icosphere_mesh(const Point3 &center,
                               float radius,
                               size_t divisions,
                               bool generateNormals,
                               bool generateUVs);
///
/// \param p1
/// \param p2
/// \param p3
/// \param p4
/// \param generateNormals
/// \param generateUVs
/// \return
RawMesh *create_quad_mesh(const Point3 &p1,
                          const Point3 &p2,
                          const Point3 &p3,
                          const Point3 &p4,
                          bool generateNormals,
                          bool generateUVs);
///
/// \param p1
/// \param p2
/// \param p3
/// \param p4
/// \param triangleFaces
/// \return
RawMesh *create_quad_wireframe_mesh(const Point3 &p1,
                                    const Point3 &p2,
                                    const Point3 &p3,
                                    const Point3 &p4,
                                    bool triangleFaces = false);
} // ponos namespace

#endif //PONOS_GEOMETRIC_SHAPES_H
