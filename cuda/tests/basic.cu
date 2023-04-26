#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include "gptl.h"
#include "gptl_cuda.h"

__global__ void setup_handles (int *, int *); // global routine initialize GPU handles
__global__ void start_timer (int);
__global__ void stop_timer (int);
__global__ void runit (int, int, int, int, int, float, bool, double *); // global routine drives GPU calculations
__global__ void dosleep_glob (int, int, float, double *);
__device__ void dosleep_dev (int, int, float, double *);

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
  bool kernelkernel = false;   // whether to employ kernel launches kernel (screws up GPTL)
  int nwarps;                  // total number warps in the computation
  int warp, warpsav;
  float sleepsec = 1.;         // default: sleep 1 sec
  double wc;                   // wallclock measured on CPU
  double *accum;               // accumulated measured sleep time per warp
  double accummax, accummin;   // max and min measured sleep times across warps
  int *total_gputime, *sleep1; // handles required for start/stop on GPU

  // Retrieve information about the GPU and set defaults
  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);
  blocksize = cores_per_sm;
  niter = cores_per_gpu;

  // Check for user-supplied input to override defaults
  while ((c = getopt (argc, argv, "b:kn:s:")) != -1) {
    switch (c) {
    case 'b':
      if ((blocksize = atoi (optarg)) < 1) {
	printf ("blocksize must be > 0 %d is invalid\n", blocksize);
	return -1;
      }
      break;
    case 'k':
      kernelkernel = true;
      break;
    case 'n':
      if ((niter = atoi (optarg)) < 1) {
	printf ("niter must be > 0 %d is invalid\n", niter);
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
      printf ("Usage: %s [-b blocksize] [-k] [-n niter] [-s sleepsec]\n", argv[0]);
      printf ("sleepsec can be fractional\n");
      return 2;
    }
  }
  oversub = (float) niter / cores_per_gpu;
  printf ("oversubscription factor=%f\n", oversub);  // diagnostic only
  printf ("kernelkernel=%d\n", (int) kernelkernel);

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

  // Ensure that all warps will be examined by GPTL
  ret = GPTLsetoption (GPTLmaxwarps_gpu, nwarps);
  
  // Use gettimeofday() as the underlying CPU timer. This is optional
  ret = GPTLsetutr (GPTLgettimeofday);

  // Initialize the GPTL library on CPU and GPU. This is mandatory before any start/stop
  ret = GPTLinitialize ();

  (void) (cudaMallocManaged ((void **) &total_gputime, sizeof (int)));
  (void) (cudaMallocManaged ((void **) &sleep1, sizeof (int)));
  (void) (cudaMallocManaged ((void **) &accum,  sizeof (double) * nwarps));

  for (warp = 0; warp < nwarps; ++warp)
    accum[warp] = 0.;
    
  // Define handles. It is globally accessed on the GPU so just 1 block and thread is needed
  setup_handles <<<1,1>>> (total_gputime, sleep1);
  cudaDeviceSynchronize ();
  
  printf ("Sleeping %f seconds on GPU\n", sleepsec);

  ret = GPTLstart ("total");  // CPU timer start

  // It is now legal for kernels to run kernels. But when "runit" runs "dosleep", the SM changes
  // between "start" and "stop" calls, resulting in vastly wrong and possibly negative delta.
  if (kernelkernel) {
    runit<<<1,1>>> (niter, nblocks, blocksize, *total_gputime, *sleep1, sleepsec,
		    kernelkernel, accum);
  } else {
    start_timer <<<1,1>>> (*total_gputime);
    runit<<<nblocks,blocksize>>> (niter, nblocks, blocksize, *total_gputime, *sleep1, sleepsec,
				  kernelkernel, accum);
    cudaDeviceSynchronize ();
    stop_timer <<<1,1>>> (*total_gputime);
  }

  cudaDeviceSynchronize ();   // Ensure the GPU has finished the kernel before returning to CPU
  ret = GPTLstop ("total");   // CPU timer stop

  ret = GPTLget_wallclock ("total", -1, &wc);
  printf ("CPU says total wallclock=%9.3f seconds\n", wc);

  accummax = 0.;
  warpsav = -1;
  for (warp = 0; warp < nwarps; ++warp) {
    if (accum[warp] > accummax) {
      accummax = accum[warp];
      warpsav = warp;
    }
  }
  printf ("Max time slept=%-12.9g at warp=%d\n", accummax, warpsav);

  accummin = 1.e36;
  warpsav = -1;
  for (warp = 0; warp < nwarps; ++warp) {
#ifdef DEBUG
    printf ("accum[%2.2d]=%-12.9g\n", warp, accum[warp]);
#endif
    if (accum[warp] < accummin) {
      accummin = accum[warp];
      warpsav = warp;
    }
  }
  printf ("Min time slept=%-12.9g at warp=%d\n", accummin, warpsav);
  
  ret = GPTLpr (0);           // Print the timing results, both for CPU and GPU

  // Since running on CPU here could instead call GPTLcudadevsync().
  cudaDeviceSynchronize ();   // Ensure printing of GPU results is complete before resetting
  ret = GPTLreset ();         // Reset CPU and GPU timers

  cudaDeviceSynchronize ();   // Ensure resetting of timers is done before finalizing
  ret = GPTLfinalize ();      // Shutdown (incl. GPU)

  cudaDeviceSynchronize ();   // Ensure any printing from GPTLfinalize_gpu is done before quitting
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
__global__ void runit (int niter, int nblocks, int blocksize, int total_gputime, int sleep1,
		       float sleepsec, bool kernelkernel, double *accum)
{
  int ret;
  
  if (kernelkernel) {
    ret = GPTLsliced_up_how ("runit");
    ret = GPTLstart_gpu (total_gputime);
    // Newer cuda revs no longer allow cudaDeviceSynchronize from __global__
    // so use __syncthreads instead
    __syncthreads ();   // Ensure the dispatched kernel has finished before timer call
    dosleep_glob<<<nblocks,blocksize>>> (niter, sleep1, sleepsec, accum);
    __syncthreads ();   // Ensure the dispatched kernel has finished before timer call
    ret = GPTLstop_gpu (total_gputime);
  } else {
    dosleep_dev (niter, sleep1, sleepsec, accum);
  }
}

__global__ void dosleep_glob (int niter, int sleep1, float sleepsec, double *accum)
{
  int ret;
  int mywarp, mythread;
  double maxsav, minsav;

  ret = GPTLsliced_up_how ("dosleep_glob");
  ret = GPTLget_warp_thread (&mywarp, &mythread);
  if (mythread < niter) {
    ret = GPTLstart_gpu (sleep1);
    ret = GPTLmy_sleep (sleepsec);
    ret = GPTLstop_gpu (sleep1);
    ret = GPTLget_wallclock_gpu (sleep1, &accum[mywarp], &maxsav, &minsav);
  }
}

__device__ void dosleep_dev (int niter, int sleep1, float sleepsec, double *accum)
{
  int ret;
  int mywarp, mythread;
  double maxsav, minsav;

  ret = GPTLsliced_up_how ("dosleep_dev");
  ret = GPTLget_warp_thread (&mywarp, &mythread);
  if (mythread < niter) {
    ret = GPTLstart_gpu (sleep1);
    ret = GPTLmy_sleep (sleepsec);
    ret = GPTLstop_gpu (sleep1);
    ret = GPTLget_wallclock_gpu (sleep1, &accum[mywarp], &maxsav, &minsav);
  }
}
