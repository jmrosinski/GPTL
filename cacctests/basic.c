#include "config.h"
#include "gptl.h"
#include "gptl_acc.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <openacc.h>
#include <cuda.h>

int main (int argc, char **argv)
{
  int ret;
  int c;    // for getopt
  int n;
  int total_gputime, sleep1; // handles required by GPTLstart_gpu and GPTLstop_gpu
  int khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu;
  int idx;
  int oversub = 1;       // default: use all cuda cores with no oversubscription
  int looplen;
  int maxwarps_gpu;      // number of warps to examine

  double maxsav, minsav;
  double accummax, accummin;

  // Retrieve information about the GPU
  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);

  // Check for user-supplied input to override defaults
  while ((c = getopt (argc, argv, "o:")) != -1) {
    switch (c) {
    case 'o':
      if ((oversub = atoi (optarg)) < 1) {
	printf ("oversub must be > 0. %d is invalid\n", oversub);
	return -1;
      }
      break;
    default:
      printf ("unknown option %c\n", c);
      printf ("Usage: %s [-o oversubscription_factor]\n", argv[0]);
      return 2;
    }
  }
  printf ("oversubscription factor=%d\n", oversub);  

  maxwarps_gpu = oversub * (cores_per_gpu / warpsize);
  printf ("max warps that will be examined=%d\n", maxwarps_gpu);
  ret = GPTLsetoption (GPTLmaxwarps_gpu, maxwarps_gpu);

  // Use gettimeofday() as the underlying CPU timer. This is optional
  ret = GPTLsetutr (GPTLgettimeofday);

  // Initialize the GPTL library on CPU and GPU. This is mandatory before any start/stop
  ret = GPTLinitialize ();
  looplen = cores_per_gpu * oversub;
  double accum[looplen];

  // Define handles
#pragma acc parallel private(ret) copyout(total_gputime, sleep1)
  {
    ret = GPTLinit_handle_gpu ("total_gputime", &total_gputime);
    ret = GPTLinit_handle_gpu ("sleep1",        &sleep1);
  }

  for (n = 0; n < looplen; ++n)
    accum[n] = 0.;

  printf ("Sleeping 1 second on GPU...\n");

  ret = GPTLstart ("total_kerneltime");
#pragma acc parallel loop private(n,ret,maxsav,minsav) copyin(total_gputime,sleep1,accum) copyout(accum)
  for (n = 0; n < looplen; ++n) {
    ret = GPTLstart_gpu (total_gputime);
    ret = GPTLstart_gpu (sleep1);
    ret = GPTLmy_sleep ((float) 1.);
    ret = GPTLstop_gpu (sleep1);
    ret = GPTLget_wallclock_gpu (sleep1, &accum[n], &maxsav, &minsav);
    ret = GPTLstop_gpu (total_gputime);
  }
  
  ret = GPTLcudadevsync ();
  ret = GPTLstop ("total_kerneltime");

  accummax = 0.;
  idx = -1;
  for (n = 0; n < looplen; ++n) {
    if (accum[n] > accummax) {
      accummax = accum[n];
      idx = n;
    }
  }
  printf ("Max time slept=%12.9f at idx=%d\n", accummax, idx);

  accummin = 1.e36;
  idx = -1;
  for (n = 0; n < looplen; n += warpsize) {
    if (accum[n] != 0. && accum[n] < accummin) {
      accummin = accum[n];
      idx = n;
    }
  }
  printf ("Min time slept=%12.9g at idx=%d\n", accummin, idx);

  ret = GPTLpr (0);
  return 0;
}
