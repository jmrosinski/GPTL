#include <unistd.h> // for sleep
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "../gptl.h"
#include "../cuda/gptl_cuda.h"
#include "./localproto.h"

__global__ void setup_handles (int *, int *, int *, int *, int *, int *);
__global__ void doalot (int, int, int, int, int, 
			float *, float *, double *,
			int *, int *, int *,
			int *, int *);

__device__ int *total_gputime;
__device__ int *donothing_handle;
__device__ int *doalot_log_handle;
__device__ int *doalot_log_inner_handle;
__device__ int *doalot_sqrt_handle;
__device__ int *doalot_sqrt_double_handle;

__device__ float *logvals;
__device__ float *sqrtvals;
__device__ double *dsqrtvals;

__host__ int persist (int mostwork, int outerlooplen, 
		      int innerlooplen, int balfact, int oversub)
{
  int blocksize, gridsize;
  int inner_parallel = 1; // parallel iteration count per outermost kernel iterator
  int ret;
  int n, nn;
  int totalwork;
  cudaEvent_t tstart, tstop;
  float dt;  
  int chunksize;
  int nchunks;
  cudaError_t cret;
  static const char *thisfunc = "persist";

  if (cudaMallocManaged (&total_gputime,             sizeof (int)) != cudaSuccess)
    printf ("cudaMallocManaged error total_gputime\n");
  if (cudaMallocManaged (&donothing_handle,          sizeof (int)) != cudaSuccess)
    printf ("cudaMallocManaged error donothing_handle\n");
  if (cudaMallocManaged (&doalot_log_handle,         sizeof (int)) != cudaSuccess)
    printf ("cudaMallocManaged error doalot_log_handle\n");
  if (cudaMallocManaged (&doalot_log_inner_handle,   sizeof (int)) != cudaSuccess)
    printf ("cudaMallocManaged error doalot_log_inner_handle\n");
  if (cudaMallocManaged (&doalot_sqrt_handle,        sizeof (int)) != cudaSuccess)
    printf ("cudaMallocManaged error doalot_sqrt_handle\n");
  if (cudaMallocManaged (&doalot_sqrt_double_handle, sizeof (int)) != cudaSuccess)
    printf ("cudaMallocManaged error doalot_sqrt_double_handle\n");

  printf ("%s: issuing cudaMallocManaged calls to hold results\n", thisfunc);
  if ((cret = cudaMallocManaged (&logvals, outerlooplen * sizeof (float))) != cudaSuccess)
    printf ("cudaMallocManaged error logvals:%s\n", cudaGetErrorString (cret));
  if (cudaMallocManaged (&sqrtvals, outerlooplen * sizeof (float)) != cudaSuccess)
    printf ("cudaMallocManaged error sqrtvals\n");
  if (cudaMallocManaged (&dsqrtvals, outerlooplen * sizeof (double)) != cudaSuccess)
    printf ("cudaMallocManaged error dsqrtvals\n");

  setup_handles <<<1,1>>> (total_gputime, donothing_handle, doalot_log_handle, 
			   doalot_log_inner_handle, doalot_sqrt_handle, doalot_sqrt_double_handle);
  cudaDeviceSynchronize();
  printf ("called cudaDeviceSynchronize 1\n");

  chunksize = MIN (GPTLcompute_chunksize (oversub, inner_parallel), outerlooplen);
  nchunks = (outerlooplen + (chunksize-1)) / chunksize;
  printf ("outerlooplen=%d broken into %d kernels of chunksize=%d\n",
	  outerlooplen, nchunks, chunksize);

  n = 0;
  for (nn = 0; nn < outerlooplen; nn += chunksize) {
    printf ("chunk=%d totalwork=%d\n", n, MIN (chunksize, outerlooplen - nn));
    ++n;
  }

  for (nn = 0; nn < outerlooplen; nn += chunksize) {
    totalwork = MIN (chunksize, outerlooplen - nn);
    blocksize = MIN (GPTLcores_per_sm, totalwork);
    gridsize = (totalwork-1) / blocksize + 1;

    ret = GPTLstart ("total_kerneltime");
    ret = GPTLstart ("donothing");
    donothing <<<gridsize, blocksize>>> (total_gputime, donothing_handle);
    cudaDeviceSynchronize();
    ret = GPTLstop ("donothing");
    ret = GPTLstop ("total_kerneltime");
  }

  for (nn = 0; nn < outerlooplen; nn += chunksize) {
    totalwork = MIN (chunksize, outerlooplen - nn);
    blocksize = MIN (GPTLcores_per_sm, totalwork);
    gridsize = (totalwork-1) / blocksize + 1;

    printf ("Invoking doalot gridsize=%d blocksize=%d\n", gridsize, blocksize);

    ret = GPTLstart ("total_kerneltime");
    ret = GPTLstart ("doalot");
    doalot <<<gridsize, blocksize>>> (nn, outerlooplen, innerlooplen, balfact, mostwork, 
				      logvals, sqrtvals, dsqrtvals,
				      total_gputime, doalot_log_handle, doalot_log_inner_handle,
				      doalot_sqrt_handle, doalot_sqrt_double_handle);
    cudaDeviceSynchronize();
    ret = GPTLstop ("doalot");
    ret = GPTLstop ("total_kerneltime");
  }

  // create events (start and stop):
  cudaEventCreate(&tstart);
  cudaEventCreate(&tstop);
  // to time region of cuda code:
  cudaEventRecord(tstart, 0); // the '0' is the stream id

  printf ("Sleeping 1 second on GPU...\n");
  ret = GPTLstart ("total_kerneltime");
  ret = GPTLstart ("sleep1ongpu");
  for (nn = 0; nn < outerlooplen; nn += chunksize) {
    totalwork = MIN (chunksize, outerlooplen - nn);
    blocksize = MIN (GPTLcores_per_sm, totalwork);
    gridsize = (totalwork-1) / blocksize + 1;
    sleep <<<gridsize, blocksize>>> (1.f, outerlooplen);
    cudaDeviceSynchronize();
  }
  ret = GPTLstop ("sleep1ongpu");
  ret = GPTLstop ("total_kerneltime");

  cudaEventRecord(tstop, 0);
  cudaEventSynchronize(tstop);              // make sure 'stop' is safe to use
  cudaEventElapsedTime(&dt, tstart, tstop); // time in ms
  printf ("Stream timer for sleep=%f seconds\n", dt*0.001);

  cudaEventDestroy (tstart);
  cudaEventDestroy (tstop);

  return 0;
}

__global__ void doalot (int nn, int outerlooplen, int innerlooplen, int balfact, int mostwork, 
			float *logvals, float *sqrtvals, double *dsqrtvals,
			int *total_gputime, int *doalot_log_handle, int *doalot_log_inner_handle,
			int *doalot_sqrt_handle, int *doalot_sqrt_double_handle)
{
  int ret;
  float factor;
  int blockId;
  int n, nnn;
  int niter;

  blockId = blockIdx.x 
    + blockIdx.y * gridDim.x 
    + gridDim.x * gridDim.y * blockIdx.z; 

  n = blockId * (blockDim.x * blockDim.y * blockDim.z)
    + (threadIdx.z * (blockDim.x * blockDim.y))
    + (threadIdx.y * blockDim.x)
    + threadIdx.x;

  nnn = nn + n;
  factor = (float) nnn / (float) (outerlooplen-1);
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

  ret = GPTLstart_gpu (*total_gputime);
  if (nnn < outerlooplen) {
    ret = GPTLstart_gpu (*doalot_log_handle);
    logvals[nnn] = doalot_log (niter, innerlooplen);
    ret = GPTLstop_gpu (*doalot_log_handle);

    logvals[nnn] = doalot_log_inner (niter, innerlooplen, doalot_log_inner_handle);

    ret = GPTLstart_gpu (*doalot_sqrt_handle);
    sqrtvals[nnn] = doalot_sqrt (niter, innerlooplen);
    ret = GPTLstop_gpu (*doalot_sqrt_handle);

    ret = GPTLstart_gpu (*doalot_sqrt_double_handle);
    dsqrtvals[nnn] = doalot_sqrt_double (niter, innerlooplen);
    ret = GPTLstop_gpu (*doalot_sqrt_double_handle);
  }
  ret = GPTLstop_gpu (*total_gputime);
}

__global__ void setup_handles (int *total_gputime, int *donothing_handle, int *doalot_log_handle, 
			       int *doalot_log_inner_handle, int *doalot_sqrt_handle, int *doalot_sqrt_double_handle)
{
  int ret;
  
  ret = GPTLinit_handle_gpu ("total_gputime",      total_gputime);
  ret = GPTLinit_handle_gpu ("donothing",          donothing_handle);
  ret = GPTLinit_handle_gpu ("doalot_log",         doalot_log_handle);
  ret = GPTLinit_handle_gpu ("doalot_log_inner",   doalot_log_inner_handle);
  ret = GPTLinit_handle_gpu ("doalot_sqrt",        doalot_sqrt_handle);
  ret = GPTLinit_handle_gpu ("doalot_sqrt_double", doalot_sqrt_double_handle);
}
