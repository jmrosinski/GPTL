#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include "gptl.h"
#include "gptl_cuda.h"

__global__ void setup_handles (int *, int *); // global routine initialize GPU handles
__global__ void runit (int *, int *, int);    // global routine drives GPU calculations
__device__ void dosleep (int, int);
int *total_gputime, *sleep1;                  // handles required for start/stop on GPU

int main (int argc, char **argv)
{
  int ret;               // return code
  int c;                 // for getopt
  int khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu;
  int oversub = 1;       // default: use all cuda cores with no oversubscription
  int blocksize;
  int maxwarps_gpu;      // number of warps to examine
  int nblocks;           // number of blocks on the GPU given blocksize and oversub

  // Retrieve information about the GPU
  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);
  blocksize = cores_per_sm / warpsize;

  // Check for user-supplied input to override defaults
  while ((c = getopt (argc, argv, "b:o:")) != -1) {
    switch (c) {
    case 'b':
      if ((blocksize = atoi (optarg)) < 1) {
	printf ("oversub must be > 0. %d is invalid\n", oversub);
	return -1;
      }
      break;
    case 'o':
      if ((oversub = atoi (optarg)) < 1) {
	printf ("oversub must be > 0. %d is invalid\n", oversub);
	return -1;
      }
      break;
    default:
      printf ("unknown option %c\n", c);
      printf ("Usage: %s [-b blocksize] [-o oversubscription_factor]\n", argv[0]);
      return 2;
    }
  }
  printf ("oversubscription factor=%d\n", oversub);  

  maxwarps_gpu = oversub * (cores_per_gpu / warpsize);
  printf ("max warps that will be examined=%d\n", maxwarps_gpu);
  // Ensure we will be timing all available warps on a filled GPU. This is optional
  ret = GPTLsetoption (GPTLmaxwarps_gpu, maxwarps_gpu);

  // Use gettimeofday() as the underlying CPU timer. This is optional
  ret = GPTLsetutr (GPTLgettimeofday);

  // Initialize the GPTL library on CPU and GPU. This is mandatory before any start/stop
  ret = GPTLinitialize ();

  (void) (cudaMallocManaged ((void **) &total_gputime, sizeof (int)));
  (void) (cudaMallocManaged ((void **) &sleep1, sizeof (int)));

  // Create the handles. It is globally accessed on the GPU so just 1 block and thread is needed
  setup_handles <<<1,1>>> (total_gputime, sleep1);
  cudaDeviceSynchronize ();
  
  nblocks = oversub * (cores_per_gpu / blocksize);

  printf ("nblocks=%d blocksize=%d\n", nblocks, blocksize);
  printf ("Sleeping 1 second on GPU\n");

  ret = GPTLstart ("total");  // CPU timer start
  runit <<<nblocks,blocksize>>> (total_gputime, sleep1, warpsize);  // Run the GPU kernel
  cudaDeviceSynchronize ();   // Ensure the GPU has finished the kernel before returning to CPU
  ret = GPTLstop ("total");   // CPU timer stop

  ret = GPTLpr (0);           // Print the timing results, both for CPU and GPU
  return 0;
}

// Setup the handles: This is done ON the GPU
__global__ void setup_handles (int *total_gputime, int *sleep1)
{
  int ret;

  ret = GPTLinit_handle_gpu ("total_gputime", total_gputime);  // Set name of timer and handle 
  ret = GPTLinit_handle_gpu ("sleep1", sleep1);                // Set name of timer and handle 
}

// This routine does the work. In this case just call a sleep routine wrapped in GPU start/stop
// GPTLmy_sleep() is a convenience routine provided by GPTL to sleep some number of seconds
__global__ void runit (int *total_gputime, int *sleep1, int warpsize)
{
  dosleep (*total_gputime, *sleep1);
}

__device__ void dosleep (int total_gputime, int sleep1)
{
  int ret;

  ret = GPTLstart_gpu (total_gputime);
  ret = GPTLstart_gpu (sleep1);
  ret = GPTLmy_sleep ((float) 1.);
  ret = GPTLstop_gpu (sleep1);
  ret = GPTLstop_gpu (total_gputime);
}
