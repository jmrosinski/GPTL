#include <stdio.h>
#include "../gptl.h"
#include "../cuda/gptl_cuda.h"
#include "./localproto.h"

int main ()
{
  int maxwarps_gpu;
  int outerlooplen;
  int innerlooplen;
  int mostwork;
  int balfact;
  int oversub;
  int cores_per_sm;
  int cores_per_gpu;
  int khz, warpsize, devnum, smcount;
  
  int ret;
  int ok;
  int ans;
  
  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);
  printf ("smcount=%d\n", smcount);
  printf ("warpsize=%d\n", warpsize);
  printf ("cores_per_gpu=%d\n", cores_per_gpu);

  do {
    printf ("maxwarps_gpu is the number of warps which will be timed\n");
    maxwarps_gpu = getval_int ("maxwarps_gpu", cores_per_gpu / warpsize);
    ok = maxwarps_gpu > 0;
    if (! ok)
      printf ("maxwarps_gpu must be positive\n");
  } while (! ok);
  printf ("maxwarps_gpu=%d\n", maxwarps_gpu);
  ret = GPTLsetoption (GPTLmaxwarps_gpu, maxwarps_gpu);

  //  ret = GPTLsetoption (GPTLmaxtimers_gpu, 100)
  //  ret = GPTLsetoption (GPTLtablesize_gpu, 32)   ! This setting gives 1 collision
  printf ("Calling GPTLinitialize\n");
  ret = GPTLinitialize ();

  do {
    printf ("mostwork ensures same work distribution regardless of warp count\n");
    mostwork = getval_int ("mostwork", 1000);
    ok = mostwork > 0;
    if (! ok)
      printf ("mostwork must be > 0\n");
  } while (! ok);

  do {
    printf ("outerlooplen is the number iterations which will be run in parallel\n");
    outerlooplen = getval_int ("outerlooplen", maxwarps_gpu * warpsize);
    ok = outerlooplen > 0;
    if (! ok)
      printf ("outerlooplen must be > 0\n");
  } while (! ok);
  printf ("outerlooplen=%d\n", outerlooplen);

  do {
    printf ("innerlooplen sets the amount of work which will be done on a single cuda core\n");
    innerlooplen = getval_int ("innerlooplen", 100);
    ok = innerlooplen > 0;
    if (! ok)
      printf ("innerlooplen must be > 0\n");
  } while (! ok);

  do {
    printf ("balfact defines load balance distribution from lowest thread to highest\n");
    balfact = getval_int ("balfact: 0=LtoR 1=balanced 2=RtoL", 1);
    ok = balfact == 0 || balfact == 1 || balfact == 2;
    if (! ok)
      printf ("valid balfact values are 0, 1, or 2\n");
  } while (! ok);

  do {
    printf ("oversub defines oversubsubscription factor\n");
    oversub = getval_int ("oversub: ", (outerlooplen + (cores_per_gpu-1)) / cores_per_gpu);
    ok = oversub > 0;
    if (! ok)
      printf ("oversub must be an integer > 0\n");
  } while (! ok);

  printf ("Enter 1 to run just sleep, anything else to run the full \"persist\" suite\n");
  (void) scanf ("%d", &ans);
  if (ans == 1) {
    sleep1 (outerlooplen, oversub);
  } else {
    persist (mostwork, outerlooplen, innerlooplen, balfact, oversub);
  }
  ret = GPTLpr (0);
}
