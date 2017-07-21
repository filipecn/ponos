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

#include "algorithm/triangulate.h"

typedef int VOID;
typedef float REAL;
extern "C" {
#include <triangle.h>
}
#include <algorithm>

namespace ponos {

void triangulate(const RawMesh *input, const MeshData *data, RawMesh *output) {
  struct triangulateio in, out;
  in.numberofpoints = input->vertexDescriptor.count;
  in.pointlist =
      (REAL *)malloc(input->vertexDescriptor.count * 2 * sizeof(REAL));
  // std::copy(input->vertices.begin(), input->vertices.end(), in.pointlist);
  printf("%lu\n", input->vertexDescriptor.count);
  for (int i = 0; i < in.numberofpoints; i++) {
    in.pointlist[i * 2 + 0] = input->vertices[i * 2 + 0];
    in.pointlist[i * 2 + 1] = input->vertices[i * 2 + 1];
    std::cout << in.pointlist[i * 2 + 0] << " " << in.pointlist[i * 2 + 1]
              << std::endl;
  }
  in.numberofpointattributes =
      data->vertexAttributes.size() ? data->vertexAttributes[0].size() : 0;
  if (in.numberofpointattributes) {
    in.pointattributelist = (float *)malloc(in.numberofpointattributes *
                                            in.numberofpoints * sizeof(float));
    for (size_t i = 0; i < data->vertexAttributes.size(); i++)
      for (size_t k = 0; k < data->vertexAttributes[i].size(); k++)
        in.pointattributelist[i * in.numberofpointattributes + k] =
            data->vertexAttributes[i][k];
  }
  if (data->vertexBoundaryMarker.size()) {
    in.pointmarkerlist = (int *)malloc(in.numberofpoints * sizeof(int));
    for (int i = 0; i < in.numberofpoints; i++)
      in.pointmarkerlist[i] = data->vertexBoundaryMarker[i];
  }
  in.trianglelist = nullptr;
  in.triangleattributelist = nullptr;
  in.trianglearealist = nullptr;
  in.numberoftriangles = 0;
  in.numberoftriangleattributes = 0;
  in.numberofcorners = 0;

  in.numberofsegments = input->meshDescriptor.count;
  if (in.numberofsegments) {
    in.segmentlist =
        (int *)malloc(input->meshDescriptor.count *
                      input->meshDescriptor.elementSize * sizeof(int));
    for (int i = 0; i < in.numberofsegments; i++) {
      in.segmentlist[i * 2 + 0] = input->indices[i * 2 + 0].vertexIndex;
      in.segmentlist[i * 2 + 1] = input->indices[i * 2 + 1].vertexIndex;
      std::cout << in.segmentlist[i * 2 + 0] << " " << in.segmentlist[i * 2 + 1]
                << std::endl;
    }
  }
  if (data->edgeBoundaryMarker.size()) {
    in.segmentmarkerlist = (int *)malloc(in.numberofsegments * sizeof(int));
    for (int i = 0; i < in.numberofsegments; i++)
      in.segmentmarkerlist[i] = data->edgeBoundaryMarker[i];
  }
  in.numberofholes = data->holes.size();
  if (in.numberofholes) {
    in.holelist = (float *)malloc(in.numberofholes * 2 * sizeof(float));
    for (int i = 0; i < in.numberofholes; i++) {
      in.holelist[i * 2 + 0] = data->holes[i].first;
      in.holelist[i * 2 + 1] = data->holes[i].second;
    }
  }
  in.numberofregions = 0;

  out.pointlist = (REAL *)NULL; /* Not needed if -N switch used. */
  /* Not needed if -N switch used or number of point attributes is zero: */
  out.pointattributelist = (REAL *)NULL;
  out.pointmarkerlist = (int *)NULL; /* Not needed if -N or -B switch used. */
  out.trianglelist = (int *)NULL;    /* Not needed if -E switch used. */
  /* Not needed if -E switch used or number of triangle attributes is zero: */
  out.triangleattributelist = (REAL *)NULL;
  out.neighborlist = (int *)NULL; /* Needed only if -n switch used. */
  /* Needed only if segments are output (-p or -c) and -P not used: */
  out.segmentlist = (int *)NULL;
  /* Needed only if segments are output (-p or -c) and -P and -B not used: */
  out.segmentmarkerlist = (int *)NULL;
  out.edgelist = (int *)NULL;       /* Needed only if -e switch used. */
  out.edgemarkerlist = (int *)NULL; /* Needed if -e used and -B not used. */

  char sw[100];
  std::strcpy(sw, "pzcVDN");
  triangulate(sw, &in, &out, nullptr);

  output->primitiveType = GeometricPrimitiveType::TRIANGLES;
  output->meshDescriptor = {3, static_cast<size_t>(out.numberoftriangles)};
  output->vertexDescriptor = {2, static_cast<size_t>(out.numberofpoints)};
  for (int i = 0; i < out.numberofpoints; i++) {
    output->addVertex({out.pointlist[i * 2 + 0], out.pointlist[i * 2 + 1]});
    std::cout << out.pointlist[i * 2 + 0] << " " << out.pointlist[i * 2 + 1]
              << std::endl;
  }
  for (int i = 0; i < out.numberoftriangles; i++) {
    for (int k = 0; k < 3; k++) {
      RawMesh::IndexData a = {out.trianglelist[i * 3 + k], 0, 0};
      std::cout << out.trianglelist[i * 3 + k] << " ";
      output->indices.emplace_back(a);
    }
    std::cout << std::endl;
  }
  output->computeBBox();
  output->splitIndexData();
  output->buildInterleavedData();
  std::cout << output->bbox.pMin.xy() << " " << output->bbox.pMax.xy()
            << std::endl;
  free(in.pointlist);
  free(in.segmentlist);
  free(in.pointmarkerlist);
}

} // ponos namespace
