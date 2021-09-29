#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include "gptl.h"
#include "gptl_cuda.h"

__global__ void setup_handles (int *, int *); // global routine initialize GPU handles
__global__ void runit (int, int, int, int, int *, int *);

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
  int *handle1, *handle2;
  int siz = 0;                 // Array size (elements)
  int bad;
  int *arr1, *arr2;            // arrays for copying
  
  // Retrieve information about the GPU and set defaults
  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);
  blocksize = cores_per_sm; // default
  niter = cores_per_gpu;    // default

  // Check for user-supplied input to override defaults
  while ((c = getopt (argc, argv, "b:n:s:")) != -1) {
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
    case 's':
      if ((siz = atoi (optarg)) < 1) {
	printf ("siz (-s <arg>) must be > 0 %d is invalid\n", siz);
	return -1;
      }
      break;
    default:
      printf ("unknown option %c\n", c);
      printf ("Usage: %s [-b blocksize] [-n niter] [-s arrsiz]\n", argv[0]);
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

  (void) (cudaMallocManaged ((void **) &handle1, sizeof (int)));
  (void) (cudaMallocManaged ((void **) &handle2, sizeof (int)));
  setup_handles <<<1,1>>> (handle1, handle2);

  (void) (cudaMallocManaged ((void **) &arr1, siz * sizeof (int)));
  (void) (cudaMallocManaged ((void **) &arr2, siz * sizeof (int)));
  for (int i = 0; i < siz; ++i)
    arr1[i] = 1;
  setup_handles <<<1,1>>> (handle1, handle2);

  cudaDeviceSynchronize ();

  runit<<<nblocks,blocksize>>> (*handle1, *handle2, siz, nblocks, arr1, arr2);
  printf ("main calling cudaDeviceSynchronize ()\n");
  cudaDeviceSynchronize ();   // Ensure the GPU has finished the kernel before returning to CPU
  printf ("main done calling cudaDeviceSynchronize ()\n");

  bad = 0;
  for (int i = 0; i < siz; ++i) {
    if (arr2[i] != 1)
      ++bad;
  }
  printf ("Found %d bad elements\n", bad);
  ret = GPTLpr (0);
  return 0;
}

__global__ void runit (int handle1, int handle2, int siz, int nblocks, int *arr1, int *arr2)
{
  int i, idx;
  int ret;
  int blocksize = blockDim.x;
  int myblock   = blockIdx.x;
  int mythread  = threadIdx.x;
  int start     = myblock*blocksize + mythread;
  float sleeptime;

  ret = GPTLstart_gpu (handle1);
  sleeptime = 0.01;
  GPTLmy_sleep (sleeptime);
  ret = GPTLstop_gpu (handle1);

  ret = GPTLstart_gpu (handle2);
  for (i = start; i < siz; i += nblocks*blocksize)
    arr2[i] = arr1[i];
  ret = GPTLstop_gpu (handle2);
  return;
}

__global__ void setup_handles (int *handle1, int *handle2)
{
  int ret;
  ret = GPTLinit_handle_gpu ("sleeptime", handle1);
  ret = GPTLinit_handle_gpu ("work", handle2);
}
