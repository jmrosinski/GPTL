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

  mostwork       = getval_int ("mostwork",       1000);
  maxthreads_gpu = getval_int ("maxthreads_gpu", 3584);
  outerlooplen   = getval_int ("outerlooplen",   maxthreads_gpu);
  printf ("outerlooplen=%d\n", outerlooplen);
  innerlooplen   = getval_int ("innerlooplen",   100);
  balfact        = getval_int ("balfact: 0=LtoR 1=balanced 2=RtoL", 1);

  ret = persist (0, mostwork, maxthreads_gpu, outerlooplen, 
		 innerlooplen, balfact);
}
