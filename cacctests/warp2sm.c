#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>
#include "gptl.h"
#include "gptl_acc.h"

int main ()
{
  int khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu;
  // int nwarps;
#define nwarps 30726
  int inner = 96;
  int outer = 10242;
  int ret;
  int total_gputime, inner_loop, outer_loop;

  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);

  ret = GPTLsetoption (GPTLmaxwarps_gpu, nwarps);
  ret = GPTLinitialize ();

#pragma acc parallel private(ret) copyout(total_gputime, inner_loop, outer_loop)
  {
    ret = GPTLinit_handle_gpu ("total_gputime", &total_gputime);
    ret = GPTLinit_handle_gpu ("inner_loop",    &inner_loop);
    ret = GPTLinit_handle_gpu ("outer_loop",    &outer_loop);
  }
  ret = GPTLcudadevsync ();

  ret = GPTLstart ("total");
#pragma acc parallel private(ret) copyin (total_gputime)
  {
    ret = GPTLstart_gpu (total_gputime);
  }

#pragma acc parallel loop private(ret) copyin(outer,inner)
  for (int n = 0; n < outer; ++n) {
    ret = GPTLstart_gpu (outer_loop);
#pragma acc loop vector
    for (int k = 0; k < inner; ++k) {
      ret = GPTLstart_gpu (inner_loop);
      ret = GPTLstop_gpu (inner_loop);
    }
    ret = GPTLstop_gpu (outer_loop);
  }
  ret = GPTLstart ("devsync");
  GPTLcudadevsync ();
  ret = GPTLstop ("devsync");

#pragma acc parallel private(ret) copyin (total_gputime)
  {
    ret = GPTLstop_gpu (total_gputime);
  }
  ret = GPTLstop ("total");
  ret = GPTLpr (0);
  return 0;
}
