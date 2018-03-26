#include <stdio.h>
#include <stdlib.h>
#include "../gptl.h"
#include "../cuda/gptl_cuda.h"
#include "./localproto.h"

__host__ int persist (int myrank, int mostwork, int maxwarps_gpu, int outerlooplen, 
		      int innerlooplen, int balfact)
{
  const int cores_per_sm = 64;
  int blocksize, gridsize;
  int ret;
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

  cudaMalloc (&logvals, outerlooplen * sizeof (float));
  cudaMalloc (&sqrtvals, outerlooplen * sizeof (float));
  printf ("%s: allocated %d elements of vals\n", thisfunc, outerlooplen);

  // Warmup without timers
  warmup <<<grid, block>>> ();
  cudaDeviceSynchronize();

  ret = GPTLstart ("gpu_fromcpu");
  ret = GPTLstart ("do_nothing_cpu");
  donothing <<<grid, block>>> ();
  cudaDeviceSynchronize();
  ret = GPTLstop ("do_nothing_cpu");
  ret = GPTLstop ("gpu_fromcpu");

  printf ("Invoking doalot grid=%d block=%d\n", gridsize, blocksize);
  printf ("outerlooplen=%d innerlooplen=%d balfact=%d mostwork=%d\n", 
	  outerlooplen, innerlooplen, balfact, mostwork);

  ret = GPTLstart ("gpu_fromcpu");
  ret = GPTLstart ("doalot_cpu");
  doalot <<<grid, block>>> (outerlooplen, innerlooplen, balfact, mostwork, logvals, sqrtvals);
  cudaDeviceSynchronize();
  ret = GPTLstop ("doalot_cpu");
  ret = GPTLstop ("gpu_fromcpu");

  ret = GPTLstart ("gpu_fromcpu");
  ret = GPTLstart ("sleep1ongpu");
  sleep1 <<<grid, block>>> (outerlooplen);
  cudaDeviceSynchronize();
  ret = GPTLstop ("sleep1ongpu");
  ret = GPTLstop ("gpu_fromcpu");

  ret = GPTLpr (myrank);
  return 0;
}

__global__ void warmup (void)
{
  int ret;
}

__global__ void donothing (void)
{
  int ret;

  ret = GPTLstart_gpu ("all_gpucalls");
  ret = GPTLstart_gpu ("do_nothing_gpu");
  ret = GPTLstop_gpu ("do_nothing_gpu");
  ret = GPTLstop_gpu ("all_gpucalls");
}

__global__ void doalot (int outerlooplen, int innerlooplen, int balfact, int mostwork, 
			float *logvals, float *sqrtvals)
{
  int ret;
  float factor;
  int blockId;
  int n;
  int niter;

  ret = GPTLstart_gpu ("all_gpucalls");

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
  }
  ret = GPTLstop_gpu ("all_gpucalls");
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
  return sum;
}

__global__ void sleep1 (int outerlooplen)
{
  int ret;
  int blockId;
  int n;

  ret = GPTLstart_gpu ("all_gpucalls");

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
  ret = GPTLstop_gpu ("all_gpucalls");
}
