#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "../gptl.h"
#include "../cuda/gptl_cuda.h"
#include "./localproto.h"

__global__ void dummy (int);
__global__ void init_handles (int *, int *);
__global__ void doalot (int, int, int, int, int, 
			float *, float *, double *, int *, int *);

__host__ int persist (int mostwork, int outerlooplen, 
		      int innerlooplen, int balfact, int oversub)
{
  int blocksize, gridsize;
  int inner_parallel = 1; // parallel iteration count per outermost kernel iterator
  int ret;
  int n, nn;
  int totalwork;
  int *handle, *handle2;  // Handles for 2 specific GPU calls
  float *logvals;
  float *sqrtvals;
  double *dsqrtvals;
  cudaEvent_t tstart, tstop;
  float dt;  
  int chunksize;
  int nchunks;
  static const char *thisfunc = "persist";

  printf ("%s: running dummy kernel 0: CUDA barfs if hashtable is no longer a valid pointer\n",
	  thisfunc);
  dummy <<<1,1>>> (0);
  cudaDeviceSynchronize();

  chunksize = MIN (GPTLcompute_chunksize (oversub, inner_parallel), outerlooplen);
  nchunks = (outerlooplen + (chunksize-1)) / chunksize;
  printf ("outerlooplen=%d broken into %d kernels of chunksize=%d\n",
	  outerlooplen, nchunks, chunksize);

  // GPU-specific init_handle routine needed because its tablesize likely differs from CPU
  cudaMalloc (&handle, sizeof (int));
  cudaMalloc (&handle2, sizeof (int));

  init_handles <<<1,1>>> (handle, handle2);
  cudaDeviceSynchronize();

  dummy <<<1,1>>> (1);
  cudaDeviceSynchronize();

  printf ("%s: issuing cudaMalloc calls to hold results\n", thisfunc);
  cudaMalloc (&logvals, outerlooplen * sizeof (float));
  cudaMalloc (&sqrtvals, outerlooplen * sizeof (float));
  cudaMalloc (&dsqrtvals, outerlooplen * sizeof (double));
  n = 0;
  for (nn = 0; nn < outerlooplen; nn += chunksize) {
    printf ("chunk=%d totalwork=%d\n", n, MIN (chunksize, outerlooplen - nn));
    ++n;
  }

  for (nn = 0; nn < outerlooplen; nn += chunksize) {
    totalwork = MIN (chunksize, outerlooplen - nn);
    blocksize = MIN (GPTLcores_per_sm, totalwork);
    gridsize = (totalwork-1) / blocksize + 1;

    printf ("%s: block=%d blocksize=%d gridsize=%d totalwork=%d\n",
	    thisfunc, nn, blocksize, gridsize, totalwork);
  
    ret = GPTLstart ("total_kerneltime");
    ret = GPTLstart ("donothing");
    donothing <<<gridsize, blocksize>>> ();
    cudaDeviceSynchronize();
    ret = GPTLstop ("donothing");
    ret = GPTLstop ("total_kerneltime");

    printf ("Invoking doalot gridsize=%d blocksize=%d\n", gridsize, blocksize);

    ret = GPTLstart ("total_kerneltime");
    ret = GPTLstart ("doalot");
    doalot <<<gridsize, blocksize>>> (nn, outerlooplen, innerlooplen, balfact, mostwork, 
				      logvals, sqrtvals, dsqrtvals, handle, handle2);
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
			float *logvals, float *sqrtvals, double *dsqrtvals, int *handle, int *handle2)
{
  int ret;
  float factor;
  int blockId;
  int n, nnn;
  int niter;

  ret = GPTLstart_gpu ("total_gputime");

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
    
  if (nnn < outerlooplen) {
    ret = GPTLstart_gpu ("doalot_log");
    logvals[nnn] = doalot_log (niter, innerlooplen);
    ret = GPTLstop_gpu ("doalot_log");

    logvals[nnn] = doalot_log_inner (niter, innerlooplen);

    ret = GPTLstart_gpu ("doalot_sqrt");
    sqrtvals[nnn] = doalot_sqrt (niter, innerlooplen);
    ret = GPTLstop_gpu ("doalot_sqrt");

    ret = GPTLstart_gpu ("doalot_sqrt_double");
    dsqrtvals[nnn] = doalot_sqrt_double (niter, innerlooplen);
    ret = GPTLstop_gpu ("doalot_sqrt_double");

    // Repeat same call from above to match Fortran which has an "_c" variant
    ret = GPTLstart_gpu ("doalot_sqrt_c");
    sqrtvals[nnn] = doalot_sqrt (niter, innerlooplen);
    ret = GPTLstop_gpu ("doalot_sqrt_c");

    ret = GPTLstart_handle_gpu ("doalot_handle_sqrt_c", handle);
    sqrtvals[nnn] = doalot_sqrt (niter, innerlooplen);
    ret = GPTLstop_handle_gpu ("doalot_handle_sqrt_c", handle);

    // Repeat same call from above to match Fortran which has an "_c" variant
    ret = GPTLstart_handle_gpu ("a", handle2);
    sqrtvals[nnn] = doalot_sqrt (niter, innerlooplen);
    ret = GPTLstop_handle_gpu ("a", handle2);
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
