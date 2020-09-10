#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <openacc.h>
#include <cuda.h>
#include "gptl.h"
#include "gptl_acc.h"

int main (int argc, char **argv)
{
  int ret;                     // return code
  int c;                       // for getopt
  int khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu;
  float oversub;               // oversubscription fraction (diagnostic)
  int niter;                   // total number of iterations: default cores_per_gpu
  int extraiter;               // number of extra iterations that don't complete a block
  int nwarps;                  // total number warps in the computation
  int warp, warpsav;
  float sleepsec = 1.;         // default: sleep 1 sec
  double wc;                   // wallclock measured on CPU
  double accummax, accummin;
  int total_gputime, sleep1;   // handles required by GPTLstart_gpu and GPTLstop_gpu

  // Retrieve information about the GPU and set defaults
  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);
  niter = cores_per_gpu;

  // Check for user-supplied input to override defaults
  while ((c = getopt (argc, argv, "n:s:")) != -1) {
    switch (c) {
    case 'n':
      if ((niter = atoi (optarg)) < 1) {
	printf ("niter must be > 0 %d is invalid\n", niter);
	return -1;
      }
      break;
    case 's':
      if ((sleepsec = atof (optarg)) < 0.) {
	printf ("sleepsec cannot be < 0. %f is invalid\n", sleepsec);
	return -1;
      }
      break;
    default:
      printf ("unknown option %c\n", c);
      printf ("Usage: %s [-n niter] [-s sleepsec]\n", argv[0]);
      printf ("sleepsec can be fractional\n");
      return 2;
    }
  }
  oversub = (float) niter / cores_per_gpu;
  printf ("oversubscription factor=%f\n", oversub);  // diagnostic only

  nwarps = niter / warpsize;
  if (niter % warpsize != 0) {
    extraiter = warpsize - (niter - (nwarps*warpsize));
    ++nwarps;
    printf ("Last iteration will be only %d elements\n", extraiter);
  }

  // Ensure that all warps will be examined by GPTL
  ret = GPTLsetoption (GPTLmaxwarps_gpu, nwarps);
  
  // Use gettimeofday() as the underlying CPU timer. This is optional
  ret = GPTLsetutr (GPTLgettimeofday);

  // Initialize the GPTL library on CPU and GPU
  ret = GPTLinitialize ();

  double accum[nwarps];
  for (warp = 0; warp < nwarps; ++warp)
    accum[warp] = 0.;

  // Define handles
#pragma acc parallel private(ret) copyout(total_gputime, sleep1)
  {
    ret = GPTLinit_handle_gpu ("total_gputime", &total_gputime);
    ret = GPTLinit_handle_gpu ("sleep1",        &sleep1);
  }
  ret = GPTLcudadevsync ();

  printf ("Sleeping %f seconds on GPU\n", sleepsec);

  ret = GPTLstart ("total");
#pragma acc parallel loop private(ret) copyin(total_gputime,sleep1,warpsize) copyout(accum)
  for (int n = 0; n < niter; ++n) {
    int mywarp, mythread;
    double maxsav, minsav;
    ret = GPTLsliced_up_how ("loop");
    ret = GPTLstart_gpu (total_gputime);
    ret = GPTLstart_gpu (sleep1);
    ret = GPTLmy_sleep (sleepsec);
    ret = GPTLstop_gpu (sleep1);
    ret = GPTLget_warp_thread (&mywarp, &mythread);
    ret = GPTLget_wallclock_gpu (sleep1, &accum[mywarp], &maxsav, &minsav);
    ret = GPTLstop_gpu (total_gputime);
  }
  ret = GPTLcudadevsync ();
  ret = GPTLstop ("total");

  ret = GPTLget_wallclock ("total", -1, &wc);
  printf ("CPU says total wallclock=%9.3f seconds\n", wc);

  accummax = 0.;
  warpsav = -1;
  for (warp = 0; warp < nwarps; ++warp) {
    if (accum[warp] > accummax) {
      accummax = accum[warp];
      warpsav = warp;
    }
  }
  printf ("Max time slept=%-12.9g at warp=%d\n", accummax, warpsav);

  accummin = 1.e36;
  warpsav = -1;
  for (warp = 0; warp < nwarps; ++warp) {
#ifdef DEBUG
    printf ("accum[%2.2d]=%-12.9g\n", warp, accum[warp]);
#endif
    if (accum[warp] < accummin) {
      accummin = accum[warp];
      warpsav = warp;
    }
  }
  printf ("Min time slept=%-12.9g at warp=%d\n", accummin, warpsav);

  ret = GPTLpr (0);
  return 0;
}
