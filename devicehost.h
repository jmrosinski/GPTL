/*
** $Id$
**
** Author: Jim Rosinski
**
** Contains definitions shared between CPU and GPU
*/

/* longest timer name allowed (probably safe to just change) */
#ifndef GPTL_DH
#define GPTL_DH

#define SUCCESS 0
#define FAILURE -1

#define MAX_CHARS 63
// Warpsize will be verified by the library
#define WARPSIZE 32
// Pascal: 56 SMs 64 cuda cores each = 3584 cores
#define DEFAULT_MAXWARPS_GPU 1792
#define DEFAULT_MAXTIMERS_GPU 30

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

extern "C" {
__host__   int GPTLinitialize_gpu (const int, const int, const int, const double);
__global__ void GPTLfinalize_gpu (void);
__global__ void GPTLenable_gpu (void);
__global__ void GPTLdisable_gpu (void);
__global__ void GPTLreset_gpu (void);
__global__ void GPTLfill_gpustats (Gpustats *, int *, int *);
__global__ void GPTLget_memstats_gpu (float *, float *);
__global__ void GPTLget_gpusizes (int *, int *);
__global__ void GPTLget_overhead_gpu (long long *,            // Getting my warp index
				      long long *,            // start/stop pair
				      long long *,            // Underlying timing routine
				      long long *,            // misc start code
				      long long *,            // misc stop code
				      long long *,            // self_ohd
				      long long *,            // parent_ohd
				      long long *,            // my_strlen ohd
				      long long *);           // STRMATCH ohd
}
#endif
