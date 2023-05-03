#include <stdio.h>
#include <cuda.h>
#include "gptl.h"
#include "gptl_cuda.h"

__global__ void setup_handles (int *); // global routine initialize GPU handles
__global__ void runit (int);           // global routine drives GPU calculations

int main (int argc, char **argv)
{
  int ret;                     // return code
  int khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu;
  int blocksize=32;            // blocksize for kernel launched
  int nblocks=2;               // number of blocks on the GPU given blocksize and oversub
  int niter;                   // total number of iterations: default cores_per_gpu
  int nwarps;                  // total number warps in the computation
  int *total_gputime;          // handles required for start/stop on GPU

  // Retrieve information about the GPU and set defaults
  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);
  blocksize = cores_per_sm;
  niter = cores_per_gpu;
  nwarps = niter / warpsize;
  nblocks = niter / blocksize;

  printf ("%d iters broken into nblocks=%d blocksize=%d\n", niter, nblocks, blocksize);
  // Initialize the GPTL library on CPU and GPU. This is mandatory before any start/stop
  ret = GPTLinitialize ();

  // Define handles. It is globally accessed on the GPU so just 1 block and thread is needed
  setup_handles <<<1,1>>> (total_gputime);
  cudaDeviceSynchronize ();
  
  ret = GPTLstart ("total");  // CPU timer start

  runit<<<nblocks,blocksize>>> (*total_gputime);
  cudaDeviceSynchronize ();

  cudaDeviceSynchronize ();   // Ensure the GPU has finished the kernel before returning to CPU
  ret = GPTLstop ("total");   // CPU timer stop

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
__global__ void setup_handles (int *total_gputime)
{
  int ret;

  ret = GPTLinit_handle_gpu ("total_gputime", total_gputime);  // Set name of timer and handle 
}

__global__ void runit (const int handle)
{
  int ret;
  
  ret = GPTLstart_gpu (handle);
  printf ("Hello from thread=%d block=%d on the GPU\n", threadIdx.x, blockIdx.x);
  ret = GPTLstop_gpu (handle);
}
