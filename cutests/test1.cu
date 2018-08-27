#include <stdio.h>
#include <cuda.h>
#include "../gptl.h"
#include "../cuda/gptl_cuda.h"

static int warpsize = -1;
static int smcount = -1;

__host__ int get_gpu_props (void);
__global__ void runit (float, float);

int main ()
{
  static int blocksize = 128;
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

  ret = get_gpu_props ();
  printf ("CUDA says warpsize=%d\n", warpsize);
  printf ("CUDA says smcount=%d\n", smcount);

  cores_per_gpu = GPTLcompute_chunksize (1, 1);
  printf ("GPTL says cores_per_gpu=%d\n", cores_per_gpu);
  do {
    printf ("Enter cores_per_gpu or -1 to accept default [%d]\n", cores_per_gpu);
    scanf ("%d", &ans);
    if (ans == -1)
      break;
    ok = ans % warpsize == 0;
    if (! ok)
      printf ("your response must divide evenly into warpsize=%d\n", warpsize);
    else
      cores_per_gpu = ans;
  } while (! ok);
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
  nthreads = nwarps * warpsize;
  nblocks = nthreads / blocksize;
  printf ("nblocks=%d blocksize=%d\n", nblocks, blocksize);
  runit <<<nblocks,blocksize>>> (sleep_tot, sleep_percall);
  cudaDeviceSynchronize ();
  ret = GPTLstop ("total");
  ret = GPTLpr (0);
  return 0;
}

__host__ int get_gpu_props (void)
{
  cudaDeviceProp prop;
  cudaError_t err;
  static const char *thisfunc = "GPTLget_gpu_props";

  if ((err = cudaGetDeviceProperties (&prop, 0)) != cudaSuccess) {
    printf ("%s: error:%s", thisfunc, cudaGetErrorString (err));
    return -1;
  }

  warpsize = prop.warpSize;
  smcount = prop.multiProcessorCount;
  return 0;
}

__global__ void runit (float sleep_tot, float sleep_percall)
{
  int ret;
  double slept = 0.;
  __shared__ double accum;
  __shared__ double maxtime, mintime;

  ret = GPTLstart_gpu ("runit");
  while (slept < sleep_tot) {
    ret = GPTLstart_gpu ("percall");
    ret = GPTLmy_sleep (sleep_percall);
    ret = GPTLstop_gpu ("percall");
    if (true) {
      slept += sleep_percall;
    } else if (threadIdx.x == 0) {
      ret = GPTLget_wallclock_gpu ("percall", &accum, &maxtime, &mintime);
      __syncthreads();
      slept += accum;
      printf ("threadIdx.x=%d slept=%f\n", threadIdx.x, slept);
    }
  }
  ret = GPTLstop_gpu ("runit");
}
