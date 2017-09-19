#include <stdio.h>
#include <stdlib.h>
#include "../gptl.h"
#include "../cuda/gptl_acc.h"

int main ()
{
  dim3 grid (10);
  dim3 block (100);
  const int nhandles = 2;
  int ret;
  int n;
  int maxthreads_gpu = 3584;
  int outerlooplen = 1000;
  int innerlooplen = 10;
  int ans;
  int handle[nhandles];
  float *vals;
  extern float doalot(), doalot2();

  vals = (float *) malloc (outerlooplen * sizeof (float));

  //JR NOTE: gptlinitialize call increases mallocable memory size on GPU. That call will fail
  //JR if any GPU activity happens before the call to gptlinitialize
  ret = GPTLsetoption (gptlmaxthreads_gpu, maxthreads_gpu);
  printf ("persist: calling gptlinitialize\n");
  ret = GPTLinitialize ();
  //JR Need to call GPU-specific init_handle routine because its tablesize may differ from CPU
  cudaMalloc ((void **) &handle, nhandles * sizeof (int));
  GPTLinit_handle_gpu <<<1,1>>> ("doalot_handle_sqrt_c", &handle[0]);
  GPTLinit_handle_gpu <<<1,1>>> ("a", &handle[1]);

  ret = GPTLstart ("doalot_cpu");
  docalcs1 <<<grid, block>>> (outerlooplen, innerlooplen);
  ret = GPTLstop ("doalot_cpu");

  ret = GPTLstart ("doalot_cpu_nogputimers");
  docalcs2 <<<grid, block>>> (outerlooplen, innerlooplen);
  ret = GPTLstop ("doalot_cpu_nogputimers");

  ret = GPTLpr (0);
}

__global__ void docalcs1 (int outerlooplen, int innerlooplen)
{
  int n;
  int ret;
  for (n = 0; n < outerlooplen; ++n) {
    ret = GPTLstart_gpu ("doalot_log");
    doalot (n, innerlooplen);
    ret = GPTLstop_gpu ("doalot_log");

    ret = GPTLstart_gpu ("doalot_sqrt");
    doalot2 (n, innerlooplen);
    ret = GPTLstop_gpu ("doalot_sqrt");
  }
}

__global__ int docalcs2 (int outerlooplen, int innerlooplen)
{
  int n;
  int ret[outerlooplen];
  int ret2[outerlooplen];
  for (n = 0; n < outerlooplen; ++n) {
    ret[n] = doalot (n, innerlooplen);
    ret2[n] = doalot2 (n, innerlooplen);
  }
  return 0;
}

__device__ int doalot (int n, int innerlooplen)
{
  int i, iter;
  float sum;

  sum = 0.;
  for (iter = 0; iter < innerlooplen; ++iter) {
    for (i = 0; i < n; ++i) {
      sum += log ((iter*i) + 1.);
    }
  }
}

__device__ int doalot2 (int n, int innerlooplen)
{
  int i, iter;
  float sum;

  sum = 0.;
  for (iter = 0; iter < innerlooplen; ++iter) {
    for (i = 0; i < n; ++i) {
      sum += sqrt ((iter*i) + 1.);
    }
  }
}
