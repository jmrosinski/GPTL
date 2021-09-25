#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include "gptl.h"
#include "gptl_cuda.h"

__global__ void setup_handles (int *); // global routine initialize GPU handles
__global__ void runit (int);

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
  int *handle;
  
  // Retrieve information about the GPU and set defaults
  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);
  blocksize = cores_per_sm; // default
  niter = cores_per_gpu;    // default

  // Check for user-supplied input to override defaults
  while ((c = getopt (argc, argv, "b:n:")) != -1) {
    switch (c) {
    case 'b':
      if ((blocksize = atoi (optarg)) < 1) {
	printf ("blocksize (-b <arg>)must be > 0 %d is invalid\n", blocksize);
	return -1;
      }
      break;
    case 'n':
      if ((niter = atoi (optarg)) < 1) {
	printf ("niter (-n <arg>) must be > 0 %d is invalid\n", niter);
	return -1;
      }
      break;
    default:
      printf ("unknown option %c\n", c);
      printf ("Usage: %s [-b blocksize] [-k] [-n niter]\n", argv[0]);
      return 2;
    }
  }
  oversub = (float) niter / cores_per_gpu;
  printf ("oversubscription factor=%f\n", oversub);  // diagnostic only
  
  nblocks = niter / blocksize;
  if (niter % blocksize != 0) {
    extraiter = niter - nblocks*blocksize;  // diagnostic only
    nblocks = niter / blocksize + 1;
  }
  printf ("%d iters broken into nblocks=%d blocksize=%d\n", niter, nblocks, blocksize);
  if (niter % blocksize != 0)
    printf ("Last iteration will be only %d elements\n", extraiter);
  
  // Initialize the GPTL library on CPU and GPU. This is mandatory before any start/stop
  ret = GPTLinitialize ();

  (void) (cudaMallocManaged ((void **) &handle, sizeof (int)));
  setup_handles <<<1,1>>> (handle);
  cudaDeviceSynchronize ();

  runit<<<nblocks,blocksize>>> (*handle);
  printf ("main calling cudaDeviceSynchronize ()\n");
  cudaDeviceSynchronize ();   // Ensure the GPU has finished the kernel before returning to CPU
  printf ("main done calling cudaDeviceSynchronize ()\n");
  ret = GPTLpr (0);
  return 0;
}

__global__ void runit (int handle)
{
  int ret;
  float sleeptime;

  ret = GPTLstart_gpu (handle);
  sleeptime = 0.01;
  GPTLmy_sleep (sleeptime);
  ret = GPTLstop_gpu (handle);
  return;
}

__global__ void setup_handles (int *handle)
{
  int ret;
  ret = GPTLinit_handle_gpu ("sleeptime", handle);
}
