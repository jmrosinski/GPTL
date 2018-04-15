#include <stdio.h>
#include <stdlib.h>
#include "../gptl.h"
#include "../cuda/gptl_cuda.h"
#include "./localproto.h"

__global__ void dummy (int);
__global__ void init_handles (int *, int *);
__global__ void donothing (void);
__global__ void doalot (int, int, int, int, 
			float *, float *, int *, int *);
__global__ void sleep1 (int);


__host__ int persist (int myrank, int mostwork, int maxwarps_gpu, int outerlooplen, 
		      int innerlooplen, int balfact)
{
  const int cores_per_sm = 64;
  int blocksize, gridsize;
  int ret;
  int *handle, *handle2;  // Handles for 2 specific GPU calls
  float *logvals;
  float *sqrtvals;
  static const char *thisfunc = "persist";

  blocksize = min (cores_per_sm, outerlooplen);
  gridsize = outerlooplen / blocksize;
  if (outerlooplen % blocksize != 0)
    ++gridsize;

  dim3 block (blocksize);
  dim3 grid (gridsize);

  printf ("%s: blocksize=%d gridsize=%d\n", thisfunc, blocksize, gridsize);
  
  //JR NOTE: gptlinitialize call increases mallocable memory size on GPU. That call will fail
  //JR if any GPU activity happens before the call to gptlinitialize
  ret = GPTLsetoption (GPTLmaxwarps_gpu, maxwarps_gpu);
  printf ("%s: calling gptlinitialize\n", thisfunc);
  ret = GPTLinitialize ();
  printf ("%s: running dummy kernel 0: CUDA will barf if hashtable is no longer a valid pointer\n",
	  thisfunc);

  dummy <<<1,1>>> (0);
  cudaDeviceSynchronize();

  // GPU-specific init_handle routine needed because its tablesize likely differs from CPU
  cudaMalloc (&handle, sizeof (int));
  cudaMalloc (&handle2, sizeof (int));
  init_handles <<<1,1>>> (handle, handle2);
  cudaDeviceSynchronize();

  dummy <<<1,1>>> (1);
  cudaDeviceSynchronize();

  printf ("%s: issuing cudaMalloc calls to hold results 1\n", thisfunc);
  cudaMalloc (&logvals, outerlooplen * sizeof (float));
  cudaMalloc (&sqrtvals, outerlooplen * sizeof (float));
  printf ("%s: allocated %d elements of vals\n", thisfunc, outerlooplen);

  ret = GPTLstart ("total_kerneltime");
  ret = GPTLstart ("donothing");

  donothing <<<grid, block>>> ();
  cudaDeviceSynchronize();

  ret = GPTLstop ("donothing");
  ret = GPTLstop ("total_kerneltime");

  printf ("Invoking doalot grid=%d block=%d\n", gridsize, blocksize);
  printf ("outerlooplen=%d innerlooplen=%d balfact=%d mostwork=%d\n", 
	  outerlooplen, innerlooplen, balfact, mostwork);

  ret = GPTLstart ("total_kerneltime");
  ret = GPTLstart ("doalot");

  doalot <<<grid, block>>> (outerlooplen, innerlooplen, balfact, mostwork, 
			    logvals, sqrtvals, handle, handle2);
  cudaDeviceSynchronize();

  ret = GPTLstop ("doalot");
  ret = GPTLstop ("total_kerneltime");

  cudaEvent_t tstart, tstop;
  float dt;

  // create events (start and stop):
  cudaEventCreate(&tstart);
  cudaEventCreate(&tstop);
  // to time region of cuda code:
  cudaEventRecord(tstart, 0); // the '0' is the stream id

  printf ("Sleeping 1 second on GPU...\n");
  ret = GPTLstart ("total_kerneltime");
  ret = GPTLstart ("sleep1ongpu");

  sleep1 <<<grid, block>>> (outerlooplen);
  cudaDeviceSynchronize();

  ret = GPTLstop ("sleep1ongpu");
  ret = GPTLstop ("total_kerneltime");

  cudaEventRecord(tstop, 0);
  cudaEventSynchronize(tstop);              // make sure 'stop' is safe to use
  cudaEventElapsedTime(&dt, tstart, tstop); // time in ms
  printf ("Stream timer for sleep=%f seconds\n", dt*0.001);

  ret = GPTLpr (myrank);
  cudaEventDestroy (tstart);
  cudaEventDestroy (tstop);

  return 0;
}

__global__ void donothing (void)
{
  int ret;

  ret = GPTLstart_gpu ("total_gputime");
  ret = GPTLstart_gpu ("donothing");
  ret = GPTLstop_gpu ("donothing");
  ret = GPTLstop_gpu ("total_gputime");
}

__global__ void doalot (int outerlooplen, int innerlooplen, int balfact, int mostwork, 
			float *logvals, float *sqrtvals, int *handle, int *handle2)
{
  int ret;
  float factor;
  int blockId;
  int n;
  int niter;

  ret = GPTLstart_gpu ("total_gputime");

  blockId = blockIdx.x 
    + blockIdx.y * gridDim.x 
    + gridDim.x * gridDim.y * blockIdx.z; 

  n = blockId * (blockDim.x * blockDim.y * blockDim.z)
    + (threadIdx.z * (blockDim.x * blockDim.y))
    + (threadIdx.y * blockDim.x)
    + threadIdx.x;

  factor = (float) n / (float) (outerlooplen-1);
  switch (balfact) {
  case 0:
    niter = (int) (factor * mostwork);
    break;
  case 1:
    niter = mostwork;
    break;
  case 2:
    niter = mostwork - (int) (factor * mostwork);
    break;
  default:
    printf ("doalot: bad balfact=%d--returning prematurely\n", balfact);
    return;
  }
    
  if (n < outerlooplen) {
    ret = GPTLstart_gpu ("doalot_log");
    logvals[n] = doalot_log (niter, innerlooplen);
    ret = GPTLstop_gpu ("doalot_log");

    logvals[n] = doalot_log_inner (niter, innerlooplen);

    ret = GPTLstart_gpu ("doalot_sqrt");
    sqrtvals[n] = doalot_sqrt (niter, innerlooplen);
    ret = GPTLstop_gpu ("doalot_sqrt");

    // Repeat same call from above to match Fortran which has an "_c" variant
    ret = GPTLstart_gpu ("doalot_sqrt_c");
    sqrtvals[n] = doalot_sqrt (niter, innerlooplen);
    ret = GPTLstop_gpu ("doalot_sqrt_c");

    ret = GPTLstart_handle_gpu ("doalot_handle_sqrt_c", handle);
    sqrtvals[n] = doalot_sqrt (niter, innerlooplen);
    ret = GPTLstop_handle_gpu ("doalot_handle_sqrt_c", handle);

    // Repeat same call from above to match Fortran which has an "_c" variant
    ret = GPTLstart_handle_gpu ("a", handle2);
    sqrtvals[n] = doalot_sqrt (niter, innerlooplen);
    ret = GPTLstop_handle_gpu ("a", handle2);
  }
  ret = GPTLstop_gpu ("total_gputime");
}

__device__ float doalot_log (int n, int innerlooplen)
{
  int i, iter;
  float sum;

  sum = 0.;
  for (iter = 0; iter < innerlooplen; ++iter) {
    for (i = 0; i < n; ++i) {
      sum += log ((iter*i) + 1.);
    }
  }
  return sum;
}

__device__ float doalot_log_inner (int n, int innerlooplen)
{
  int i, iter;
  int ret;
  float sum;

  sum = 0.;
  for (iter = 0; iter < innerlooplen; ++iter) {
    ret = GPTLstart_gpu ("doalot_log_inner");
    for (i = 0; i < n; ++i) {
      sum += log ((iter*i) + 1.);
    }
    ret = GPTLstop_gpu ("doalot_log_inner");
  }
  return sum;
}

__device__ float doalot_sqrt (int n, int innerlooplen)
{
  int i, iter;
  float sum;

  sum = 0.;
  for (iter = 0; iter < innerlooplen; ++iter) {
    for (i = 0; i < n; ++i) {
      sum += sqrt ((iter*i) + 1.);
    }
  }
  // Add bogus never-execute printf to force compiler not to optimize out
  // due to multiple calls from doalot
  if (sum < 0.00001)
    printf ("sum=%f\n", sum);
  return sum;
}

__global__ void sleep1 (int outerlooplen)
{
  int ret;
  int blockId;
  int n;

  ret = GPTLstart_gpu ("total_gputime");

  blockId = blockIdx.x 
    + blockIdx.y * gridDim.x 
    + gridDim.x * gridDim.y * blockIdx.z; 

  n = blockId * (blockDim.x * blockDim.y * blockDim.z)
    + (threadIdx.z * (blockDim.x * blockDim.y))
    + (threadIdx.y * blockDim.x)
    + threadIdx.x;

  if (n < outerlooplen) {
    ret = GPTLstart_gpu ("sleep1");
    ret = GPTLmy_sleep (1.);
    ret = GPTLstop_gpu ("sleep1");
  }
  ret = GPTLstop_gpu ("total_gputime");
}

__global__ void dummy (int id)
{
  GPTLdummy_gpu (id);
}

__global__ void init_handles (int *handle, int *handle2)
{
  int ret;

  ret = GPTLinit_handle_gpu ("doalot_handle_sqrt_c", handle);
  ret = GPTLinit_handle_gpu ("a", handle2);
}
