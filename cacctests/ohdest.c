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
  int khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu;
  const int niter_cpu = 1000;  // number of CPU iterations (outer loop)
  int niter;                   // number of GPU iterations: cores_per_gpu
  int nwarps;                  // number of warps in each GPU calc. (cores_per_gpu/warpsize)
  int warp, warpsav;
  double wc_timed;             // wallclock for timed GPU loop (measured on CPU)
  double wc_untimed;           // wallclock for untimed GPU loop (measured on CPU)
  double wc_devsync;           // wallclock for GPTLcudadevsync ();
  double ohdest;               // overhead estimate for single start/stop pair on GPU
  double accummax, accummin;   // max/min time taken across GPU warps
  // GPU handles:
  int timed_loop;

  // Retrieve information about the GPU and set defaults
  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);
  niter = cores_per_gpu;
  nwarps = niter / warpsize;
  if (niter % warpsize != 0) {
    printf ("Something is wrong: niter=%d mod warpsize=%d not zero\n", niter, warpsize);
    return 1;
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
#pragma acc parallel private(ret) copyout(timed_loop)
  {
    ret = GPTLinit_handle_gpu ("timed_loop", &timed_loop);
  }
  ret = GPTLcudadevsync ();

  ret = GPTLstart ("total");
  // Loop a bunch of times on CPU to get non-zero times
  for (int nn = 0; nn < niter_cpu; ++nn) {
    // First: call start and stop on GPU and time on CPU
    ret = GPTLstart ("timed_loop");
#pragma acc parallel loop private(ret) copyin(timed_loop)
    for (int n = 0; n < niter; ++n) {
      ret = GPTLstart_gpu (timed_loop);
      ret = GPTLstop_gpu (timed_loop);
    }
    ret = GPTLcudadevsync ();
    ret = GPTLstop ("timed_loop");

    // Next: do nothing on GPU and time on CPU. Do the same copyin as for untimed loop
    // even though not needed, to obtain a fair time estimate. This cost is substantial.
    ret = GPTLstart ("untimed_loop");
#pragma acc parallel loop copyin(timed_loop)
    for (int n = 0; n < niter; ++n) {
      GPTLdummy_gpu ();
    }
    ret = GPTLcudadevsync ();
    ret = GPTLstop ("untimed_loop");

    // Finally: Estimate cost of GPTLcudadevsync() (hopefully small);
    ret = GPTLstart ("cudadevsync");
    ret = GPTLcudadevsync ();
    ret = GPTLstop ("cudadevsync");
  }

  ret = GPTLstop ("total");

  // Next: discover time spread across warps for timed loop 
#pragma acc parallel loop private(ret) copyout(accum)
  for (int n = 0; n < niter; ++n) {
    int mywarp, mythread;
    double maxsav, minsav;
    ret = GPTLget_warp_thread (&mywarp, &mythread);
    ret = GPTLget_wallclock_gpu (timed_loop, &accum[mywarp], &maxsav, &minsav);
  }

  ret = GPTLcudadevsync ();
  accummax = 0.;
  warpsav = -1;
  for (warp = 0; warp < nwarps; ++warp) {
    if (accum[warp] > accummax) {
      accummax = accum[warp];
      warpsav = warp;
    }
  }
  printf ("Max time=%-12.9g seconds at warp=%d\n", accummax, warpsav);

  accummin = 1.e36;
  warpsav = -1;
  for (warp = 0; warp < nwarps; ++warp) {
    if (accum[warp] < accummin) {
      accummin = accum[warp];
      warpsav = warp;
    }
  }
  printf ("Min time=%-12.9g seconds at warp=%d\n", accummin, warpsav);

  // Finally: Estimate overhead as GPU start/stop minus GPU do-nothing
  ret = GPTLget_wallclock ("timed_loop",   -1, &wc_timed);
  ret = GPTLget_wallclock ("untimed_loop", -1, &wc_untimed);
  ret = GPTLget_wallclock ("cudadevsync",  -1, &wc_devsync);

  printf ("timed GPU loop took   %g seconds\n", wc_timed);
  printf ("untimed GPU loop took %g seconds\n", wc_untimed);
  printf ("GPTLcudadevsync took  %g seconds\n", wc_devsync);

  ohdest = (wc_timed - wc_untimed) /  niter_cpu;
  printf ("Overhead estimate of single GPU start/stop pair=%g usec\n", ohdest * 1.e6);
  printf ("See timing.000000 for overall results\n");
  ret = GPTLpr (0);
  return 0;
}
