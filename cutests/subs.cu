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

__global__ void donothing (void)
{
  int ret;

  ret = GPTLstart_gpu ("total_gputime");
  ret = GPTLstart_gpu ("total_gputime2");
  ret = GPTLstart_gpu ("donothing");
  ret = GPTLstop_gpu ("donothing");
  ret = GPTLstop_gpu ("total_gputime2");
  ret = GPTLstop_gpu ("total_gputime");
}
