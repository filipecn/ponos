#include <gtest/gtest.h>
#include <ponos.h>

#pragma GCC diagnostic ignored "-Wwrite-strings"
#ifdef SINGLE
#define REAL float
#else /* not SINGLE */
#define REAL double
#endif /* not SINGLE */
typedef int VOID;

extern "C" {
#include <triangle.h>
}

#include <stdio.h>
#include <stdlib.h>

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

TEST(Triangulate, triangulate) {
  struct triangulateio in, mid; //, out; //, vorout;

  /* Define input points. */

  in.numberofpoints = 4;
  in.numberofpointattributes = 0;
  in.pointlist = (REAL *)malloc(in.numberofpoints * 2 * sizeof(REAL));
  in.pointlist[0] = 0.0;
  in.pointlist[1] = 0.0;
  in.pointlist[2] = 1.0;
  in.pointlist[3] = 0.0;
  in.pointlist[4] = 1.0;
  in.pointlist[5] = 1.0;
  in.pointlist[6] = 0.0;
  in.pointlist[7] = 1.0;
  in.pointattributelist = (REAL *)NULL;
  in.pointmarkerlist = (int *)malloc(in.numberofpoints * sizeof(int));
  in.pointmarkerlist[0] = 1;
  in.pointmarkerlist[1] = 1;
  in.pointmarkerlist[2] = 1;
  in.pointmarkerlist[3] = 1;

  in.numberofsegments = 4;
  in.segmentlist = (int *)malloc(in.numberofsegments * 2 * sizeof(int));
  in.segmentlist[0] = 0;
  in.segmentlist[1] = 1;
  in.segmentlist[2] = 1;
  in.segmentlist[3] = 2;
  in.segmentlist[4] = 2;
  in.segmentlist[5] = 3;
  in.segmentlist[6] = 3;
  in.segmentlist[7] = 0;
  in.segmentmarkerlist = (int *)malloc(in.numberofpoints * sizeof(int));
  in.segmentmarkerlist[0] = 1;
  in.segmentmarkerlist[1] = 1;
  in.segmentmarkerlist[2] = 1;
  in.segmentmarkerlist[3] = 1;
  in.numberofholes = 0;
  in.numberofregions = 0;
  in.regionlist = (REAL *)NULL;

  printf("Input point set:\n\n");
  report(&in, 1, 0, 0, 0, 0, 0);

  mid.pointlist = (REAL *)NULL; /* Not needed if -N switch used. */
  mid.pointattributelist = (REAL *)NULL;
  mid.pointmarkerlist = (int *)NULL; /* Not needed if -N or -B switch used. */
  mid.trianglelist = (int *)NULL;    /* Not needed if -E switch used. */
  mid.triangleattributelist = (REAL *)NULL;
  mid.neighborlist = (int *)NULL; /* Needed only if -n switch used. */
  mid.segmentlist = (int *)NULL;
  mid.segmentmarkerlist = (int *)NULL;
  mid.edgelist = (int *)NULL;       /* Needed only if -e switch used. */
  mid.edgemarkerlist = (int *)NULL; /* Needed if -e used and -B not used. */

  triangulate("pzYAen", &in, &mid, (struct triangulateio *)NULL);
  printf("Initial triangulation:\n\n");
  report(&mid, 1, 1, 1, 1, 1, 0);
}
