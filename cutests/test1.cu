#include <stdio.h>
#include <cuda.h>
#include "../gptl.h"
#include "../cuda/gptl_cuda.h"

__global__ void runit (float, float, int *, int *);
__global__ void setup_handles (int *, int *);

__device__ int *runit_handle;
__device__ int *percall_handle;

int main ()
{
  static int blocksize = 128;
  int warpsize = -1;
  int khz = -1;
  int devnum = -1;
  int smcount = -1;
  int cores_per_sm = -1;
  int cores_per_gpu = -1;
  int oversub = -1;
  int nwarps;
  int nthreads;
  int nblocks;
  int ans;
  int ok;
  
  int ret;
  float sleep_tot;
  float sleep_percall;

  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);
  printf ("warpsize=%d\n",      warpsize);
  printf ("smcount=%d\n",       smcount);
  printf ("cores_per_sm=%d\n",  cores_per_sm);
  printf ("cores_per_gpu=%d\n", cores_per_gpu);

  printf ("Enter oversubsubscription factor\n");
  scanf ("%d", &oversub);
  printf ("oversub=%d\n", oversub);

  nwarps = (cores_per_gpu * oversub) / warpsize;
  printf ("nwarps=%d\n", nwarps);
  if (nwarps * warpsize != cores_per_gpu * oversub)
    printf ("NOTE: warpsize=%d does not divide evenly into cores_per_gpu(%d) * oversub(%d)=%d\n",
	    warpsize, cores_per_gpu, oversub, cores_per_gpu * oversub);
  ret = GPTLsetoption (GPTLmaxwarps_gpu, nwarps);

  printf ("Enter sleep_tot sleep_percall (both in floating point seconds)\n");
  scanf ("%f%f", &sleep_tot, &sleep_percall);
  printf ("sleep_tot=%f sec sleep_percall=%f sec\n", sleep_tot, sleep_percall);

  ret = GPTLinitialize ();
  ret = GPTLstart ("total");

  (void) (cudaMallocManaged (&runit_handle,   sizeof (int)));
  (void) (cudaMallocManaged (&percall_handle, sizeof (int)));

  setup_handles <<<1,1>>> (runit_handle, percall_handle);
  cudaDeviceSynchronize ();
  
  nthreads = nwarps * warpsize;
  nblocks = nthreads / blocksize;

  printf ("nblocks=%d blocksize=%d\n", nblocks, blocksize);
  runit <<<nblocks,blocksize>>> (sleep_tot, sleep_percall, runit_handle, percall_handle);
  cudaDeviceSynchronize ();
  ret = GPTLstop ("total");
  ret = GPTLpr (0);
  return 0;
}

__global__ void setup_handles (int *runit_handle, int *percall_handle)
{
  int ret;
  
  ret = GPTLinit_handle_gpu ("runit",   runit_handle);
  ret = GPTLinit_handle_gpu ("percall", percall_handle);
}

__global__ void runit (float sleep_tot, float sleep_percall, int *runit_handle, int *percall_handle)
{
  int ret;
  double slept = 0.;
  __shared__ double accum;
  __shared__ double maxtime, mintime;

  ret = GPTLstart_gpu (*runit_handle);

  while (slept < sleep_tot) {
    ret = GPTLstart_gpu (*percall_handle);
    ret = GPTLmy_sleep (sleep_percall);
    ret = GPTLstop_gpu (*percall_handle);
    if (true) {
      slept += sleep_percall;
    } else if (threadIdx.x == 0) {
      ret = GPTLget_wallclock_gpu (*percall_handle, &accum, &maxtime, &mintime);
      __syncthreads();
      slept += accum;
      printf ("threadIdx.x=%d slept=%f\n", threadIdx.x, slept);
    }
  }
  ret = GPTLstop_gpu (*runit_handle);
}
