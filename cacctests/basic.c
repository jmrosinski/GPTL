#include "config.h"
#include <gptl.h>
#include <gptl_acc.h>
#include <stdio.h>
#include <stdlib.h>

int main ()
{
  int ret;
  int n;
  int total_gputime, sleep1;  //need cudaMallocManaged for these
  int khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu;
  int idx;

  double maxsav, minsav;
  double *accum;
  double accumsav;

#pragma acc routine (GPTLinit_handle_gpu) seq
  ret = GPTLinitialize ();
  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);
  accum = (double *) malloc (cores_per_gpu);

  // Define handles
#pragma acc parallel private(ret)
  {
    ret = GPTLinit_handle_gpu ("total_gputime", &total_gputime);
    ret = GPTLinit_handle_gpu ("sleep1",        &sleep1);
  }

  printf ("Sleeping 1 second on GPU...\n");
  for (n = 0; n < cores_per_gpu; ++n)
    accum[n] = 0.;
#pragma acc data copy (accum)
  {
    ret = GPTLstart ("total_kerneltime");
#pragma acc parallel loop private(ret,maxsav,minsav) copyin(total_gputime,sleep1)
    for (n = 0; n < cores_per_gpu; ++n) {
      ret = GPTLstart_gpu (total_gputime);
      ret = GPTLstart_gpu (sleep1);
      ret = GPTLmy_sleep ((float) 1.);
      ret = GPTLstop_gpu (sleep1);
      ret = GPTLget_wallclock_gpu (sleep1, &accum[n], &maxsav, &minsav);
      ret = GPTLstop_gpu (total_gputime);
    }
    ret = GPTLcudadevsync ();
  }
  ret = GPTLstop ("total_kerneltime");

  accumsav = 0.;
  idx = -1;
  for (n = 0; n < cores_per_gpu; ++n) {
    if (accum[n] > accumsav) {
      accumsav = accum[n];
      idx = n;
    }
  }
  printf ("Max time slept=%g at idx=%d\n", accumsav, idx);

  ret = GPTLpr (0);
  return 0;
}
