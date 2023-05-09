#include <stdio.h>
#include <cuda.h>
#include "gptl.h"
#include "gptl_cuda.h"

__global__ void setup_handle (int *);   // global routine initialize GPU handles
__global__ void startstop (int, int *); // global routine calls GPTL start and stop

int main (int argc, char **argv)
{
  int ret;                     // return code
  int khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu;
  int blocksize;               // blocksize for kernel launched
  int nblocks;                 // number of blocks on the GPU given blocksize and oversub
  int *handle;                 // handle for start/stop
  int *global_retcode;         // "return code" from global routine

  // Retrieve information about the GPU and set defaults
  ret       = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);
  blocksize = cores_per_sm;
  nblocks   = smcount;
  
  // Verify GPTLinitialize returns success
  if ((ret = GPTLinitialize ()) != 0) {
    printf ("GPTLinitialize returns %d when it should have returned 0\n", ret);
    return -1;
  }

  // Create shared CPU/GPU space for handle to be used in start/stop
  // And for "return code" from global routine
  (void) (cudaMallocManaged ((void **) &handle, sizeof (int)));
  (void) (cudaMallocManaged ((void **) &global_retcode, sizeof (int)));

  // Create a single handle for timing
  setup_handle <<<1,1>>> (handle);
  cudaDeviceSynchronize ();

  // Verify all start/stop calls from within a kernel succeed
  *global_retcode = 0;
  startstop <<<nblocks, blocksize>>> (*handle, global_retcode);
  cudaDeviceSynchronize ();
  
  if (*global_retcode != 0) {
    printf ("One or more start/stop pairs failed. global_retcode=%d\n", *global_retcode);
    return -1;
  }
  
  if ((ret = GPTLfinalize ()) != 0) {
    printf ("GPTLfinalize returns %d when it should have returned 0\n", ret);
    return -1;
  }
  return 0;
}

// Setup the handles: This is done ON the GPU
__global__ void setup_handle (int *handle)
{
  GPTLinit_handle_gpu ("some_timer", handle);
}

// Call start and stop and change global_retcode if any single call fails
__global__ void startstop (int handle, int *global_retcode)
{
  int ret;
  if ((ret = GPTLstart_gpu (handle)) != 0) {
    *global_retcode = ret;
    if ((ret = GPTLstop_gpu (handle)) != 0) {
      *global_retcode = ret;
    }
  }
}
