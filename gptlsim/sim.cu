#include <stdio.h>
#include <cuda.h>
#include "./proto.h"

#define SUCCESS 0
#define WARPSIZE 32
#define NOT_ROOT_OF_WARP -2
#define WARPID_GT_MAXWARPS -3

__device__ static Timer *table;
__device__ static int maxwarps;
__device__ static int *badaddress = 0;

__device__ static int start (void);
__device__ static int stop (void);
__device__ static inline int get_warp_num ();

__global__ void init_sim (Timer *table_cpu, int maxwarps_cpu)
{
  int i;
  
  table = table_cpu;
  maxwarps = maxwarps_cpu;
  for (i = 0; i < maxwarps_cpu; ++i) {
    table[i].start = 0;
    table[i].stop = 0;
  }
}

__global__ void run_sim (void)
{
  int ret;
  int iter = 0;
  
  while (1) {
    ret = start ();
    ret = stop ();
    ++iter;
  }
}

__device__ static int start (void)
{
  int w;
  long long stamp;
  long long delta;
  
  w = get_warp_num ();

  // Return if not thread 0 of the warp, or warpId is outside range of available timers
  if (w == NOT_ROOT_OF_WARP || w == WARPID_GT_MAXWARPS)
    return SUCCESS;

  stamp = clock64 ();
  delta = stamp - table[w].stop;
  if (delta < 0) {
    printf ("start ERROR FOUND: w=%d stop=%lld stamp=%lld\n", w, table[w].stop, stamp);
    w = *badaddress;
  }
  table[w].start = stamp;
  return SUCCESS;
}

__device__ static int stop (void)
{
  int w;
  long long stamp;
  long long delta;
  
  w = get_warp_num ();

  // Return if not thread 0 of the warp, or warpId is outside range of available timers
  if (w == NOT_ROOT_OF_WARP || w == WARPID_GT_MAXWARPS)
    return SUCCESS;

  stamp = clock64 ();
  delta = stamp - table[w].start;
  if (delta < 0) {
    printf ("stop ERROR FOUND: w=%d start=%lld stamp=%lld\n", w, table[w].start, stamp);
    w = *badaddress;
  }
  table[w].stop = stamp;
  return SUCCESS;
}

__device__ static inline int get_warp_num ()
{
  int warpId;
  int blockId = blockIdx.x 
    + blockIdx.y * gridDim.x 
    + gridDim.x * gridDim.y * blockIdx.z; 
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
    + (threadIdx.z * (blockDim.x * blockDim.y))
    + (threadIdx.y * blockDim.x)
    + threadIdx.x;

  // Only thread 0 of the warp will be timed
  if (threadId % WARPSIZE != 0)
    return NOT_ROOT_OF_WARP;

  // USE_THREADS means use threadId not warpId 
  warpId = threadId / WARPSIZE;

  if (warpId > maxwarps-1)
    return WARPID_GT_MAXWARPS;

  return warpId;
}
