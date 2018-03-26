#include <stdio.h>
#include "./localproto.h"

int main ()
{
  int maxwarps_gpu;
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
    maxwarps_gpu = getval_int ("maxwarps_gpu", 112);
    ok = maxwarps_gpu > 0;
    if (! ok)
      printf ("maxwarps_gpu must be positive\n");
  } while (! ok);
  maxthreads_gpu = maxwarps_gpu * 32;

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

  ret = persist (0, mostwork, maxwarps_gpu, outerlooplen, 
		 innerlooplen, balfact);
}
