/*
** $Id: private.h,v 1.74 2011-03-28 20:55:19 rosinski Exp $
**
** Author: Jim Rosinski
**
** Contains definitions private to GPTL and inaccessible to invoking user environment
*/

#ifndef GPTL_DEVICE_H
#define GPTL_DEVICE_H

#include <stdio.h>
// Need cuda.h for cudaGetErrorString() below
#include <cuda.h>
#include "devicehost.h"

#define SUCCESS 0
#define FAILURE -1
#define NOT_ROOT_OF_WARP -2
#define WARPID_GT_MAXWARPS -3
#ifdef TIME_GPTL
#define NUM_INTERNAL_TIMERS 3
#endif

#define STRMATCH(X,Y) (my_strcmp((X),(Y)) == 0)

typedef struct {
  long long last;              // timestamp from last call
  long long accum;             // accumulated time
  long long max;               // longest time for start/stop pair
  long long min;               // shortest time for start/stop pair
} Wallstats;

typedef struct TIMER {
  Wallstats wall;              // wallclock stats
  unsigned long count;         // number of start/stop calls
#ifdef ENABLE_GPURECURSION
  uint recurselvl;             // recursion level
#endif
  uint smid;                   // SM the region is running on
  uint badsmid_count;          // number of times SM id changed
  uint negdelta_count;         // number of times a negative time increment occurred
  bool onflg;                  // timer currently on or off
} Timer;

typedef struct {
  char name[MAX_CHARS+1];
} Timername;

typedef struct {
  long long accum_max;
  long long accum_min;
  long long max;
  long long min;
  unsigned long count;
  unsigned int negdelta_count_max;
  int negdelta_count_max_warp;
  unsigned int negdelta_nwarps;
  int accum_max_warp;
  int accum_min_warp;
  int nwarps;
  int count_max;
  int count_max_warp;
  int count_min;
  int count_min_warp;
  uint badsmid_count;
  char name[MAX_CHARS+1];
} Gpustats;

// Function prototypes
extern "C" {
__global__ void GPTLreset_gpu (const int, int *global_retval);
__global__ void GPTLreset_all_gpu (int *global_retval);
__global__ void GPTLfinalize_gpu (int *global_retval);
__global__ void GPTLfill_gpustats (Gpustats *, int *, int *);
__global__ void GPTLget_memstats_gpu (float *, float *);
__global__ void GPTLget_maxwarpid_info (int *, int *);
__global__ void GPTLget_overhead_gpu (float *,            // Getting my warp index
				      float *,            // start/stop pair
				      float *,            // Underlying timing routine
				      float *,            // misc start code
				      float *,            // misc stop code
				      float *,            // self_ohd
				      float *,            // parent_ohd
				      float *,            // my_strlen ohd
				      float *,            // STRMATCH ohd
				      int *);             // return code from __global__
__device__ int GPTLget_maxwarpid_timed (void);
__device__ int GPTLerror_1s (const char *, const char *);
__device__ int GPTLerror_2s (const char *, const char *, const char *);
__device__ int GPTLerror_3s (const char *, const char *, const char *, const char *);
__device__ int GPTLerror_1s1d (const char *, const char *, const int);
__device__ int GPTLerror_2s1d (const char *, const char *, const char *, const int);
__device__ int GPTLerror_2s2d (const char *, const char *, const char *, const int, const int);
__device__ int GPTLerror_2s3d (const char *, const char *, const char *, const int, const int,
			       const int);
__device__ int GPTLerror_1s2d (const char *, const char *, const int, const int);
__device__ int GPTLerror_1s1d1s (const char *, const char *, const int, const char *);
__device__ void GPTLreset_errors_gpu (void);                  /* num_errors to zero */
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
  inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
  {
    if (code != cudaSuccess) 
      {
	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
      }
  }
#endif
