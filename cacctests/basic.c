#include <stdio.h>
#include <openacc.h>
#include "gptl.h"
#include "gptl_acc.h"

int main (int argc, char **argv)
{
  int ret;                     // return code
  int khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu;
  int niter;                   // total number of iterations: default cores_per_gpu
  int nwarps;                  // total number warps in the computation
  int total_gputime;

  // Retrieve information about the GPU and set defaults
  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);
  niter = cores_per_gpu;
  nwarps = niter / warpsize;

  // Initialize the GPTL library on CPU and GPU
  ret = GPTLinitialize ();

  // Define handles
#pragma acc parallel private(ret) copyout(total_gputime)
  {
    ret = GPTLinit_handle_gpu ("total_gputime", &total_gputime);
  }

  ret = GPTLstart ("total");
#pragma acc parallel loop private(ret) copyin(total_gputime,warpsize)
  for (int n = 0; n < niter; ++n) {
    ret = GPTLstart_gpu (total_gputime);
    ret = GPTLstop_gpu (total_gputime);
  }
  ret = GPTLstop ("total");
  ret = GPTLpr (0);
  return 0;
}
