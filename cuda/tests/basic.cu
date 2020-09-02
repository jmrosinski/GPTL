#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include "gptl.h"
#include "gptl_cuda.h"

__global__ void setup_handles (int *, int *); // global routine initialize GPU handles
__global__ void start_timer (int);
__global__ void stop_timer (int);
__global__ void runit (int, int, int, int, float, bool); // global routine drives GPU calculations
__global__ void dosleep_glob (int, float);
__device__ void dosleep_dev (int, float);
__device__ void whereami (void);
int *total_gputime, *sleep1;                  // handles required for start/stop on GPU
__device__ double *accum;

int main (int argc, char **argv)
{
  int ret;               // return code
  int c;                 // for getopt
  int khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu;
  float oversub = 1.;    // default: use all cuda cores with no oversubscription
  float sleepsec = 1.;   // default: sleep 1 sec
  int maxwarps_gpu;      // number of warps to examine
  int blocksize;
  int nblocks;           // number of blocks on the GPU given blocksize and oversub
  bool kernelkernel = false;

  // Retrieve information about the GPU
  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);
  blocksize = cores_per_sm / warpsize;

  // Check for user-supplied input to override defaults
  while ((c = getopt (argc, argv, "b:k:o:s:")) != -1) {
    switch (c) {
    case 'b':
      if ((blocksize = atoi (optarg)) < 1) {
	printf ("oversub must be > 0. %f is invalid\n", oversub);
	return -1;
      }
      break;
    case 'k':
      kernelkernel = true;
      break;
    case 'o':
      if ((oversub = atof (optarg)) <= 0.) {
	printf ("oversub must be > 0. %f is invalid\n", oversub);
	return -1;
      }
      break;
    case 's':
      if ((sleepsec = atof (optarg)) < 0.) {
	printf ("sleepsec cannot be < 0. %f is invalid\n", sleepsec);
	return -1;
      }
      break;
    default:
      printf ("unknown option %c\n", c);
      printf ("Usage: %s [-b blocksize] [-k] [-o oversub] [-s sleepsec]\n", argv[0]);
      printf ("oversub and sleepsec can be fractions\n");
      return 2;
    }
  }
  printf ("oversubscription factor=%f\n", oversub);  
  printf ("kernelkernel=%d\n", (int) kernelkernel);
  
  maxwarps_gpu = oversub * (cores_per_gpu / warpsize);
  printf ("max warps that will be examined=%d\n", maxwarps_gpu);
  // Ensure we will be timing all available warps on a filled GPU. This is optional
  ret = GPTLsetoption (GPTLmaxwarps_gpu, maxwarps_gpu);

  // Use gettimeofday() as the underlying CPU timer. This is optional
  ret = GPTLsetutr (GPTLgettimeofday);

  // Initialize the GPTL library on CPU and GPU. This is mandatory before any start/stop
  ret = GPTLinitialize ();

  nblocks = oversub * (cores_per_gpu / blocksize);

  printf ("nblocks=%d blocksize=%d\n", nblocks, blocksize);
  printf ("Sleeping %f seconds on GPU\n", sleepsec);

  (void) (cudaMallocManaged ((void **) &total_gputime, sizeof (int)));
  (void) (cudaMallocManaged ((void **) &sleep1, sizeof (int)));
  (void) (cudaMallocManaged ((void **) &accum,  sizeof (int) * nblocks * blocksize));

  // Create the handles. It is globally accessed on the GPU so just 1 block and thread is needed
  setup_handles <<<1,1>>> (total_gputime, sleep1);
  cudaDeviceSynchronize ();
  
  ret = GPTLstart ("total");  // CPU timer start

  // It is now legal for kernels to run kernels. But when "runit" runs "dosleep", the SM changes
  // between "start" and "stop" calls, resulting in vastly wrong and possibly negative delta.
  if (kernelkernel) {
    runit<<<1,1>>> (nblocks, blocksize, *total_gputime, *sleep1, sleepsec, kernelkernel);
  } else {
    start_timer <<<1,1>>> (*total_gputime);
    runit<<<nblocks,blocksize>>> (nblocks, blocksize, *total_gputime, *sleep1, sleepsec, kernelkernel);
    stop_timer <<<1,1>>> (*total_gputime);
  }
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

__global__ void start_timer (int handle)
{
  int ret = GPTLstart_gpu (handle);
}

__global__ void stop_timer (int handle)
{
  int ret = GPTLstop_gpu (handle);
}

// This routine does the work. In this case just call a sleep routine wrapped in GPU start/stop
// GPTLmy_sleep() is a convenience routine provided by GPTL to sleep some number of seconds
__global__ void runit (int nblocks, int blocksize, int total_gputime, int sleep1,
		       float sleepsec, bool kernelkernel)
{
  int ret;
  if (kernelkernel) {
    ret = GPTLstart_gpu (total_gputime);
    dosleep_glob<<<nblocks,blocksize>>> (sleep1, sleepsec);
    cudaDeviceSynchronize ();   // Ensure the dispatched kernel has finished before timer call
    ret = GPTLstop_gpu (total_gputime);
  } else {
    dosleep_dev (sleep1, sleepsec);
  }
}

__global__ void dosleep_glob (int sleep1, float sleepsec)
{
  int ret;

  ret = GPTLstart_gpu (sleep1);
  ret = GPTLmy_sleep (sleepsec);
  ret = GPTLstop_gpu (sleep1);
  whereami ();
}

__device__ void dosleep_dev (int sleep1, float sleepsec)
{
  int ret;

  ret = GPTLstart_gpu (sleep1);
  ret = GPTLmy_sleep (sleepsec);
  ret = GPTLstop_gpu (sleep1);
  whereami ();
}

__device__ void whereami ()
{
  int warpid;
  int threadid = threadIdx.x
    +  blockDim.x  * threadIdx.y
    +  blockDim.x  *  blockDim.y  * threadIdx.z
    +  blockDim.x  *  blockDim.y  *  blockDim.z  * blockIdx.x
    +  blockDim.x  *  blockDim.y  *  blockDim.z  *  gridDim.x  * blockIdx.y
    +  blockDim.x  *  blockDim.y  *  blockDim.z  *  gridDim.x  *  gridDim.y  * blockIdx.z;

  if (threadid % 32 == 0) {
    warpid = threadid / 32;
    printf ("threadid=%d warpid=%d\n", threadid, warpid);
    printf ("threadIdx.x=%d\n", threadIdx.x);
    if (threadIdx.y > 0)
      printf ("threadIdx.y=%d added=%d\n",
	      threadIdx.y, blockDim.x*threadIdx.y);
    if (threadIdx.z > 0)
      printf ("threadIdx.z=%d added=%d\n",
	      threadIdx.z, blockDim.x*blockDim.y*threadIdx.z);
    if (blockIdx.x > 0)
      printf ("blockIdx.x=%d added=%d\n",
	      blockIdx.x, blockDim.x*blockDim.y*blockDim.z*blockIdx.x);
    if (blockIdx.y > 0)
      printf ("blockIdx.y=%d added=%d\n",
	      blockIdx.y, blockDim.x*blockDim.y*blockDim.z*gridDim.x*blockIdx.y);
    if (blockIdx.z > 0)
      printf ("blockIdx.z=%d added=%d\n",
	      blockIdx.z, blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y*blockIdx.y);
  }
}
