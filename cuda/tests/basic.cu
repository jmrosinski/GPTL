#include <stdio.h>
#include <cuda.h>
#include "gptl.h"
#include "gptl_cuda.h"

__global__ void runit (float, int *);  // global routine drives GPU calculations
__global__ void setup_handles (int *); // global routine initialize GPU handles
__device__ int *sleep_handle;          // handle required for start/stop on GPU

int main ()
{
  static int blocksize = 128;  // hardwire to something any GPU can handle
  // These things inited to -1 come back from GPTLget_gpu_props()
  int warpsize = -1;
  int khz = -1;
  int devnum = -1;
  int smcount = -1;
  int cores_per_sm = -1;
  int cores_per_gpu = -1;
  int nwarps;            // number of warps on the GPU
  int nblocks;           // number of blocks on the GPU given blocksize
  
  int ret;               // return code
  float sleeptime = 1.;  // sleep this long and time it

  // Retrieve information about the GPU
  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);

  nwarps = (cores_per_gpu) / warpsize;
  printf ("Total warps on GPU=nwarps=%d\n", nwarps);
  // Ensure we will be timing all available warps on a filled GPU. This is optional
  ret = GPTLsetoption (GPTLmaxwarps_gpu, nwarps);
  // Use gettimeofday() as the underlying CPU timer. This is optional
  ret = GPTLsetutr (GPTLgettimeofday);
  // Initialize the GPTL library on CPU and GPU. This is mandatory before any start/stop
  ret = GPTLinitialize ();

  // Simplest to use managed memory to create the GPU timing handles (only one used here)
  (void) (cudaMallocManaged (&sleep_handle, sizeof (int)));
  // Create the handle. It is globally accessed on the GPU so just 1 block and thread is needed
  setup_handles <<<1,1>>> (sleep_handle);
  cudaDeviceSynchronize ();
  
  nblocks = (nwarps * warpsize) / blocksize;

  printf ("nblocks=%d blocksize=%d\n", nblocks, blocksize);
  printf ("Sleeping %f seconds and timing it\n", sleeptime);

  ret = GPTLstart ("total");  // CPU timer start
  runit <<<nblocks,blocksize>>> (sleeptime, sleep_handle);  // Run the GPU kernel
  cudaDeviceSynchronize ();   // Ensure the GPU has finished the kernel before returning to CPU
  ret = GPTLstop ("total");   // CPU timer stop
  ret = GPTLpr (0);           // Print the timing results, both for CPU and GPU
  return 0;
}

// Setup the handles: This is done ON the GPU
__global__ void setup_handles (int *sleep_handle)
{
  int ret = GPTLinit_handle_gpu ("sleep", sleep_handle);  // Set name of timer and handle 
}

// This routine does the work. In this case just call a sleep routine wrapped in GPU start/stop
// GPTLmy_sleep() is a convenience routine provided by GPTL to sleep some number of seconds
__global__ void runit (float sleeptime, int *sleep_handle)
{
  int ret;

  ret = GPTLstart_gpu (*sleep_handle);
  ret = GPTLmy_sleep (sleeptime);
  ret = GPTLstop_gpu (*sleep_handle);
}
