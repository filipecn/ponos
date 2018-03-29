/*
 * Copyright (c) 2017 FilipeCN
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

#ifndef AERGIA_SCENE_MESH_UTILS_H
#define AERGIA_SCENE_MESH_UTILS_H

#include <ponos/ponos.h>

namespace aergia {

/// \param d **[in]** dimensions
/// \param s **[in]** scale
/// \param o **[in]** offset
/// \return pointer to a <RawMesh> object with the list of vertices and indices
/// that describe a grid in the 3D space.
ponos::RawMesh *create_grid_mesh(const ponos::ivec3 &d, float s,
                                 const ponos::vec3 &o);
/// \param m **[in]** base mesh
/// \return RawMesh representing the edges of **m**.
ponos::RawMesh *create_wireframe_mesh(const ponos::RawMesh *m);
/// Generates an icosphere's triangle mesh
/// \param center center
/// \param radius radius
/// \param divisions subdivision number
/// \param generateNormals include normals
/// \param genereateUVs include texture coordinates
/// \return RawMesh representing the icosphere
ponos::RawMesh *create_icosphere_mesh(const ponos::Point3 &center,
                                      float radius,
                                      size_t divisions,
                                      bool generateNormals = false,
                                      bool genereateUVs = false);

} // aergia namespace

#endif // AERGIA_SCENE_MESH_UTILS_H
