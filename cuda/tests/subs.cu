#include <cuda.h>
#include <math.h>
#include "../cuda/gptl_cuda.h"
#include "./localproto.h"

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

__device__ float doalot_log_inner (int n, int innerlooplen, int *handle)
{
  int i, iter;
  int ret;
  float sum;

  sum = 0.;
  for (iter = 0; iter < innerlooplen; ++iter) {
    ret = GPTLstart_gpu (*handle);
    for (i = 0; i < n; ++i) {
      sum += log ((iter*i) + 1.);
    }
    ret = GPTLstop_gpu (*handle);
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
      sum += sqrtf ((float) iter*i);
    }
  }
  return sum;
}

__device__ double doalot_sqrt_double (int n, int innerlooplen)
{
  int i, iter;
  double sum;

  sum = 0.;
  for (iter = 0; iter < innerlooplen; ++iter) {
    for (i = 0; i < n; ++i) {
      sum += sqrt ((double) iter*i);
    }
  }
  return sum;
}

__global__ void donothing (int *total_gputime, int *donothing_handle)
{
  int ret;

  ret = GPTLstart_gpu (*total_gputime);
  ret = GPTLstart_gpu (*donothing_handle);
  ret = GPTLstop_gpu (*donothing_handle);
  ret = GPTLstop_gpu (*total_gputime);
}
