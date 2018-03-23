#include <stdio.h>
#include "./localproto.h"

int main ()
{
  int maxthreads_gpu;
  int outerlooplen;
  int innerlooplen;
  int mostwork;
  int balfact;
  int ret;
  int ok;

  do {
    mostwork = getval_int ("mostwork",       1000);
    ok = mostwork > 0;
    if (! ok)
      printf ("mostwork must be > 0\n");
  } while (! ok);

  do {
    maxthreads_gpu = getval_int ("maxthreads_gpu", 3584);
    ok = maxthreads_gpu % 32 == 0;
    if (! ok)
      printf ("maxthreads_gpu must be a multiple of warpsize (32)\n");
  } while (! ok);

  do {
    outerlooplen   = getval_int ("outerlooplen",   maxthreads_gpu);
    ok = outerlooplen > 0;
    if (! ok)
      printf ("outerlooplen must be > 0\n");
  } while (! ok);
  printf ("outerlooplen=%d\n", outerlooplen);

  do {
    innerlooplen   = getval_int ("innerlooplen",   100);
    ok = innerlooplen > 0;
    if (! ok)
      printf ("innerlooplen must be > 0\n");
  } while (! ok);

  do {
    balfact      = getval_int ("balfact: 0=LtoR 1=balanced 2=RtoL", 1);
    ok = balfact == 0 || balfact == 1 || balfact == 2;
    if (! ok)
      printf ("valid balfact values are 0, 1, or 2\n");
  } while (! ok);

  ret = persist (0, mostwork, maxthreads_gpu, outerlooplen, 
		 innerlooplen, balfact);
}
