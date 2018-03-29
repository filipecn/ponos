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

#pragma GCC diagnostic ignored "-Wwrite-strings"

#include <algorithm>
#include <ponos/algorithm/triangulate.h>

typedef int VOID;
typedef double REAL;

#ifdef TRIANGLE_INCLUDED
extern "C" {
#include <triangle.h>
}
#endif

#ifdef TETGEN_INCLUDED
#ifndef TETLIBRARY
#define TETLIBRARY
#endif
#include <tetgen.h>
#endif

namespace ponos {

#ifdef TRIANGLE_INCLUDED
void report(struct triangulateio *io, int markers, int reporttriangles,
            int reportneighbors, int reportsegments, int reportedges,
            int reportnorms) {
  int i, j;

  for (i = 0; i < io->numberofpoints; i++) {
    printf("Point %4d:", i);
    for (j = 0; j < 2; j++) {
      printf("  %.6g", io->pointlist[i * 2 + j]);
    }
    if (io->numberofpointattributes > 0) {
      printf("   attributes");
    }
    for (j = 0; j < io->numberofpointattributes; j++) {
      printf("  %.6g",
             io->pointattributelist[i * io->numberofpointattributes + j]);
    }
    if (markers) {
      printf("   marker %d\n", io->pointmarkerlist[i]);
    } else {
      printf("\n");
    }
  }
  printf("\n");

  if (reporttriangles || reportneighbors) {
    for (i = 0; i < io->numberoftriangles; i++) {
      if (reporttriangles) {
        printf("Triangle %4d points:", i);
        for (j = 0; j < io->numberofcorners; j++) {
          printf("  %4d", io->trianglelist[i * io->numberofcorners + j]);
        }
        if (io->numberoftriangleattributes > 0) {
          printf("   attributes");
        }
        for (j = 0; j < io->numberoftriangleattributes; j++) {
          printf("  %.6g",
                 io->triangleattributelist[i * io->numberoftriangleattributes +
                     j]);
        }
        printf("\n");
      }
      if (reportneighbors) {
        printf("Triangle %4d neighbors:", i);
        for (j = 0; j < 3; j++) {
          printf("  %4d", io->neighborlist[i * 3 + j]);
        }
        printf("\n");
      }
    }
    printf("\n");
  }

  if (reportsegments) {
    for (i = 0; i < io->numberofsegments; i++) {
      printf("Segment %4d points:", i);
      for (j = 0; j < 2; j++) {
        printf("  %4d", io->segmentlist[i * 2 + j]);
      }
      if (markers) {
        printf("   marker %d\n", io->segmentmarkerlist[i]);
      } else {
        printf("\n");
      }
    }
    printf("\n");
  }

  if (reportedges) {
    for (i = 0; i < io->numberofedges; i++) {
      printf("Edge %4d points:", i);
      for (j = 0; j < 2; j++) {
        printf("  %4d", io->edgelist[i * 2 + j]);
      }
      if (reportnorms && (io->edgelist[i * 2 + 1] == -1)) {
        for (j = 0; j < 2; j++) {
          printf("  %.6g", io->normlist[i * 2 + j]);
        }
      }
      if (markers) {
        printf("   marker %d\n", io->edgemarkerlist[i]);
      } else {
        printf("\n");
      }
    }
    printf("\n");
  }
}
#endif
void triangulate(const RawMesh *input, const MeshData *data, RawMesh *output) {
#ifdef TRIANGLE_INCLUDED
  UNUSED_VARIABLE(output);
  {
    FILE *fp = fopen("D.poly", "w+");
    fprintf(fp, "%lu 2 0 1\n", input->positionDescriptor.count);
    for (size_t i = 0; i < input->positionDescriptor.count; i++)
      fprintf(fp, "%lu %f %f %d\n", i, input->positions[i * 2],
              input->positions[i * 2 + 1], data->vertexBoundaryMarker[i]);
    fprintf(fp, "%lu 0\n", input->meshDescriptor.count);
    for (size_t i = 0; i < input->meshDescriptor.count; i++)
      fprintf(fp, "%lu %d %d\n", i, input->indices[i * 2].positionIndex,
              input->indices[i * 2 + 1].positionIndex);
    fprintf(fp, "0\n");
    fclose(fp);
  }
  // struct triangulateio in, out;

  struct triangulateio in, mid; //, out, vorout;

  in.numberofpoints = 4;
  in.numberofpointattributes = 0;
  in.pointlist = (REAL *) malloc(in.numberofpoints * 2 * sizeof(REAL));
  in.pointlist[0] = 0.0;
  in.pointlist[1] = 0.0;
  in.pointlist[2] = 1.0;
  in.pointlist[3] = 0.0;
  in.pointlist[4] = 1.0;
  in.pointlist[5] = 1.0;
  in.pointlist[6] = 0.0;
  in.pointlist[7] = 1.0;
  in.pointattributelist = (REAL *) NULL;
  in.pointmarkerlist = (int *) malloc(in.numberofpoints * sizeof(int));
  in.pointmarkerlist[0] = 1;
  in.pointmarkerlist[1] = 1;
  in.pointmarkerlist[2] = 1;
  in.pointmarkerlist[3] = 1;

  in.numberofsegments = 4;
  in.segmentlist = (int *) malloc(in.numberofsegments * 2 * sizeof(int));
  in.segmentlist[0] = 0;
  in.segmentlist[1] = 1;
  in.segmentlist[2] = 1;
  in.segmentlist[3] = 2;
  in.segmentlist[4] = 2;
  in.segmentlist[5] = 3;
  in.segmentlist[6] = 3;
  in.segmentlist[7] = 0;
  in.segmentmarkerlist = (int *) malloc(in.numberofpoints * sizeof(int));
  in.segmentmarkerlist[0] = 1;
  in.segmentmarkerlist[1] = 1;
  in.segmentmarkerlist[2] = 1;
  in.segmentmarkerlist[3] = 1;
  in.numberofholes = 0;
  in.numberofregions = 0;
  in.regionlist = (REAL *) NULL;

  printf("Input point set:\n\n");
  report(&in, 1, 0, 0, 0, 0, 0);

  mid.pointlist = (REAL *) NULL; /* Not needed if -N switch used. */
  mid.pointattributelist = (REAL *) NULL;
  mid.pointmarkerlist = (int *) NULL; /* Not needed if -N or -B switch used. */
  mid.trianglelist = (int *) NULL;    /* Not needed if -E switch used. */
  mid.triangleattributelist = (REAL *) NULL;
  mid.neighborlist = (int *) NULL; /* Needed only if -n switch used. */
  mid.segmentlist = (int *) NULL;
  mid.segmentmarkerlist = (int *) NULL;
  mid.edgelist = (int *) NULL;       /* Needed only if -e switch used. */
  mid.edgemarkerlist = (int *) NULL; /* Needed if -e used and -B not used. */

  // triangulate("pzYAen", &in, &mid, (struct triangulateio *)NULL);
  printf("Initial triangulation:\n\n");
  report(&mid, 1, 1, 1, 1, 1, 0);
  /*
in.numberofpoints = input->vertexDescriptor.count_;
in.pointlist = (REAL *)malloc(in.numberofpoints * 2 * sizeof(REAL));
for (int i = 0; i < in.numberofpoints; i++) {
in.pointlist[i * 2 + 0] = input->vertices[i * 2 + 0];
in.pointlist[i * 2 + 1] = input->vertices[i * 2 + 1];
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
// if (data->vertexBoundaryMarker.size()) {
in.pointmarkerlist = (int *)malloc(in.numberofpoints * sizeof(int));
for (int i = 0; i < in.numberofpoints; i++)
in.pointmarkerlist[i] = 1; // data->vertexBoundaryMarker[i];
//}
in.trianglelist = (int *)NULL;
in.triangleattributelist = (REAL *)NULL;
in.trianglearealist = (REAL *)NULL;
in.numberoftriangles = 0;
in.numberoftriangleattributes = 0;
in.numberofcorners = 0;
in.numberofsegments = input->meshDescriptor.count_;
if (in.numberofsegments) {
in.segmentlist = (int *)malloc(in.numberofsegments * 2 * sizeof(int));
for (int i = 0; i < in.numberofsegments; i++) {
in.segmentlist[i * 2 + 0] = input->indices[i * 2 + 0].vertexIndex;
in.segmentlist[i * 2 + 1] = input->indices[i * 2 + 1].vertexIndex;
}
}
// if (data->edgeBoundaryMarker.size()) {
in.segmentmarkerlist = (int *)malloc(in.numberofsegments * sizeof(int));
for (int i = 0; i < in.numberofsegments; i++)
in.segmentmarkerlist[i] = 1; // data->edgeBoundaryMarker[i];
//}
in.numberofholes = data->holes.size();
if (in.numberofholes) {
in.holelist = (float *)malloc(in.numberofholes * 2 * sizeof(float));
for (int i = 0; i < in.numberofholes; i++) {
in.holelist[i * 2 + 0] = data->holes[i].first;
in.holelist[i * 2 + 1] = data->holes[i].second;
}
}
in.numberofregions = 0;

printf("Input point set:\n\n");
report(&in, 1, 1, 1, 1, 1, 1);

out.numberofpoints = 0;
out.pointlist = (REAL *)NULL;
out.pointattributelist = (REAL *)NULL;
out.pointmarkerlist = (int *)NULL;
out.numberoftriangleattributes = 0;
out.triangleattributelist = (REAL *)NULL;
out.numberoftriangles = 0;
out.trianglelist = (int *)NULL;
out.triangleattributelist = (REAL *)NULL;
out.neighborlist = (int *)NULL;
out.numberofsegments = 0;
out.segmentlist = (int *)NULL;
out.segmentmarkerlist = (int *)NULL;
out.numberofedges = 0;
out.edgelist = (int *)NULL;
out.edgemarkerlist = (int *)NULL;
printf("Output point set:\n\n");
report(&out, 1, 1, 1, 1, 0, 0);
char sw[256];
// std::strcpy(sw, "YBNz");
snprintf(sw, 256, "YBNz");
triangulate(sw, &in, &out, (triangulateio *)NULL);
printf("After:\n\n");
// report(&out, 1, 1, 1, 1, 0, 0);

output->primitiveType = GeometricPrimitiveType::TRIANGLES;
output->meshDescriptor = {3, static_cast<size_t>(out.numberoftriangles)};
output->vertexDescriptor = {2, static_cast<size_t>(in.numberofpoints)};
for (int i = 0; i < in.numberofpoints; i++) {
  output->addVertex({in.pointlist[i * 2 + 0], in.pointlist[i * 2 + 1]});
}
for (int i = 0; i < out.numberoftriangles; i++) {
  for (int k = 0; k < 3; k++) {
    RawMesh::IndexData a = {out.trianglelist[i * 3 + k], 0, 0};
    output->indices.emplace_back(a);
  }
}
output->computeBBox();
output->splitIndexData();
output->buildInterleavedData();
free(in.pointlist);
free(in.segmentlist);
free(in.pointmarkerlist);
      */
#else
  UNUSED_VARIABLE(input);
  UNUSED_VARIABLE(data);
  UNUSED_VARIABLE(output);
#endif
}

void tetrahedralize(const RawMesh *input, RawMesh *output) {
#ifdef TETGEN_INCLUDED
  FATAL_ASSERT(input->primitiveType == GeometricPrimitiveType::TRIANGLES);
  // add vertices
  tetgenio in, out;
  in.mesh_dim = 3;
  in.firstnumber = 0;
  in.numberofpoints = static_cast<int>(input->positionDescriptor.count);
  in.pointlist = new REAL[in.numberofpoints * in.mesh_dim];
  for (size_t i = 0; i < input->positions.size(); i++)
    in.pointlist[i] = input->positions[i];
  in.numberoffacets = static_cast<int>(input->meshDescriptor.count);
  in.facetlist = new tetgenio::facet[in.numberoffacets];
  in.facetmarkerlist = new int[in.numberoffacets];
  for (size_t i = 0; i < input->meshDescriptor.count; i++) {
    auto f = &in.facetlist[i];
    f->numberofpolygons = 1;
    f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
    f->numberofholes = 0;
    f->holelist = nullptr;
    auto p = &f->polygonlist[0];
    p->numberofvertices = 3;
    p->vertexlist = new int[p->numberofvertices];
    for (size_t k = 0; k < 3u; k++)
      p->vertexlist[k] = input->indices[i * input->meshDescriptor.elementSize + k].positionIndex;
    in.facetmarkerlist[i] = 1;
  }
  char params[20];
  strcpy(params, "VYa0.05qCMz");
  try {
    tetrahedralize(params, &in, &out);
  } catch(int e) {
    std::cerr << e << std::endl;
  }
  // set fields
  output->positionDescriptor.elementSize = input->positionDescriptor.elementSize;
  output->positionDescriptor.count = static_cast<size_t>(out.numberofpoints);
  output->meshDescriptor.elementSize = 4;
  output->meshDescriptor.count = static_cast<size_t>(out.numberoftetrahedra);
  for (int i = 0; i < out.numberofpoints * 3; i++)
    output->positions.emplace_back(out.pointlist[i]);
  for (int i = 0; i < out.numberoftetrahedra; i++)
    for (int k = 0; k < 4; k++)
      output->addFace({out.tetrahedronlist[i * 4 + k]});
  output->primitiveType = GeometricPrimitiveType::TETRAHEDRA;
  DUMP_VECTOR(input->positions);
  DUMP_VECTOR(output->positions);
#else
  UNUSED_VARIABLE(input);
  UNUSED_VARIABLE(output);
#endif
}

} // ponos namespace
