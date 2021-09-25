#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include "gptl.h"
#include "gptl_cuda.h"

__global__ void runit (int *, int, int, int, int *);    // global routine drives GPU calculations
__device__ int get_shared_idx (int, int, int *, uint);
__device__ int find_shared_idx (int, int);
__device__ void get_mutex (volatile int *);
__device__ void free_mutex (volatile int *);

#define WARPS_PER_SM 4
#define MAX_OVERSUB 1
#define SHARED_LOCS_PER_SM (WARPS_PER_SM * MAX_OVERSUB)
__shared__ volatile int timer[SHARED_LOCS_PER_SM];
__device__ volatile int mutex1[5] = {0,0,0,0,0};
__device__ volatile int mutex2[5] = {0,0,0,0,0};

typedef struct {
  int warp;
  bool inuse;
} Map;
__shared__ volatile Map map[SHARED_LOCS_PER_SM];

int main (int argc, char **argv)
{
  int ret;                     // return code
  int c;                       // for getopt
  int khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu;
  float oversub;               // oversubscription fraction (diagnostic)
  int blocksize;               // blocksize for kernel launched
  int nblocks;                 // number of blocks on the GPU given blocksize and oversub
  int niter;                   // total number of iterations: default cores_per_gpu
  int extraiter = 0;           // number of extra iterations that don't complete a block
  int nwarps;                  // total number warps in the computation
  int warps_per_sm;
  int shared_locs_per_sm;
  int *global_timer;
  int nbad;
  int *maxidx;                 // diagnostic: max index used
  
  // Retrieve information about the GPU and set defaults
  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);
  warps_per_sm = cores_per_sm / warpsize;
  blocksize = cores_per_sm; // default
  niter = cores_per_gpu;    // default

  // Check for user-supplied input to override defaults
  while ((c = getopt (argc, argv, "b:n:")) != -1) {
    switch (c) {
    case 'b':
      if ((blocksize = atoi (optarg)) < 1) {
	printf ("blocksize (-b <arg>)must be > 0 %d is invalid\n", blocksize);
	return -1;
      }
      break;
    case 'n':
      if ((niter = atoi (optarg)) < 1) {
	printf ("niter (-n <arg>) must be > 0 %d is invalid\n", niter);
	return -1;
      }
      break;
    default:
      printf ("unknown option %c\n", c);
      printf ("Usage: %s [-b blocksize] [-k] [-n niter]\n", argv[0]);
      return 2;
    }
  }
  oversub = (float) niter / cores_per_gpu;
  printf ("oversubscription factor=%f\n", oversub);  // diagnostic only
  if (oversub > MAX_OVERSUB)
    printf ("WARNING: oversub=%f MAX_OVERSUB=%d\n", oversub, MAX_OVERSUB);
  shared_locs_per_sm = warps_per_sm * MAX_OVERSUB;
  
  nwarps = niter / warpsize;
  if (niter % warpsize != 0)
    nwarps = niter / warpsize + 1;
  
  nblocks = niter / blocksize;
  if (niter % blocksize != 0) {
    extraiter = niter - nblocks*blocksize;  // diagnostic only
    nblocks = niter / blocksize + 1;
  }
  printf ("%d iters broken into nblocks=%d blocksize=%d\n", niter, nblocks, blocksize);
  if (niter % blocksize != 0)
    printf ("Last iteration will be only %d elements\n", extraiter);
  
  // Initialize the GPTL library on CPU and GPU. This is mandatory before any start/stop
  ret = GPTLinitialize ();
  
  (void) (cudaMallocManaged ((void **) &global_timer,  sizeof (int) * nwarps));
  (void) (cudaMallocManaged ((void **) &maxidx,  sizeof (int)));  // diagnostic
  for (int n = 0; n < nwarps; ++n)
    global_timer[n] = 0;

  runit<<<nblocks,blocksize>>> (global_timer, shared_locs_per_sm, warpsize, nwarps, maxidx);
  printf ("main calling cudaDeviceSynchronize ()\n");
  cudaDeviceSynchronize ();   // Ensure the GPU has finished the kernel before returning to CPU
  printf ("main done calling cudaDeviceSynchronize ()\n");

  printf ("Final value of global_timer: should be 1 everywhere\n");
  nbad = 0;
  for (int n = 0; n < nwarps; ++n)
    if (global_timer[n] != 1)
      ++nbad;
  printf ("Number of bad values in global_timer=%d\n", nbad);
  printf ("maxidx=%d\n", *maxidx);
      
  return 0;
}

__device__ void get_mutex (volatile int *mutex)
{
 bool isSet;

 do {
   // If mutex is 0, grab by setting = 1
   // If mutex is 1, it stays 1 and isSet will be false
   isSet = atomicCAS ((int *) mutex, 0, 1) == 0;
 } while ( !isSet);   // exit the loop after critical section executed
}
 
__device__ void free_mutex (volatile int *mutex)
{
  *mutex = 0;
}
 
// This routine will be called at start of GPTLstart_gpu
__device__ int get_shared_idx (int mywarp, int shared_locs_per_sm, int *maxidx, uint smid)
{
  int idx;
  int tot;

  // Use critical section to get index into shared array
  get_mutex (&mutex1[smid]);
  for (idx = 0; idx < shared_locs_per_sm; ++idx) {
    if ( ! map[idx].inuse) {
      map[idx].inuse = true;
      map[idx].warp = mywarp;
      if (idx > *maxidx)
	*maxidx = idx;  // diagnostic: max index used
      break;
    }
  }
  if (idx == shared_locs_per_sm) {
    printf ("mywarp %d no spots available!\n", mywarp);
    free_mutex (&mutex1[smid]);
    return idx;
  }
  free_mutex (&mutex1[smid]);
  return idx;
}

// This routine will be called at start of GPTLstop_gpu
__device__ int find_shared_idx (int mywarp, int shared_locs_per_sm)
{
  int n;
  int idx;

  for (idx = 0; idx < shared_locs_per_sm; ++idx) {
    if (map[idx].inuse && map[idx].warp == mywarp) {
      break;
    }
  }
  return idx;
}

__global__ void runit (int *global_timer, int shared_locs_per_sm, int warpsize, int nwarps,
		       int *maxidx)
{
  int mywarp, mythread;
  int idx;
  int idxsave;
  int ret;
  float sleeptime;
  uint smid;
  bool isSet;

  ret = GPTLget_warp_thread (&mywarp, &mythread);
  if (mythread % warpsize == 0) {
    // Stuff done in GPTLstart_gpu
    asm ("mov.u32 %0, %smid;" : "=r"(smid));
    idx = get_shared_idx (mywarp, shared_locs_per_sm, maxidx, smid);  // grab shared mem slot
    if (idx == shared_locs_per_sm) {
      printf ("runit: no slots available\n");
      return;
    }
    printf ("runit mywarp %d smid %d got idx %d\n", mywarp, smid, idx);
    timer[idx] = global_timer[mywarp];

    // Stuff done in GPTLstop_gpu
    idxsave = idx;
    idx = find_shared_idx (mywarp, shared_locs_per_sm);
    if (idx == shared_locs_per_sm) {
      printf ("runit: mywarp %d smid %d couldn't find my slot got inuse by warp %d\n",
	      mywarp, smid, map[idxsave].warp);
      return;
    }

    // Serialize the work should force use of all available idx values
    //get_mutex (&mutex2[smid]);
    sleeptime = 0.1;
    sleeptime = 0.0001 * mywarp;
    sleeptime = 0.01;
    GPTLmy_sleep (sleeptime);
    ++timer[idx];
    global_timer[mywarp] = timer[idx];

    // Reset shared memory to zero. Otherwise get random hangs due to shared memory cannot be
    // guaranteed initialized to zero.
    map[idx].inuse = false;
    map[idx].warp = 0;
    if (false)
      printf ("mywarp %d released idx %d\n", mywarp, idx);
    //free_mutex (&mutex2[smid]);
  }
  return;
}
