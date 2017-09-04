#include <stdio.h>

typedef enum {false = 0, true = 1} bool;  /* mimic C++ */
static bool verbose = false;           /* output verbosity */
extern int sub1 (bool, int);
#pragma acc routine (sub1) seq 

int sub1_ (void)
{
  int ret;

  verbose = true;
  printf ("sub1_: calling sub1\n");
#pragma acc kernels copyout(ret)
  ret = sub1 (verbose, 35);
  printf ("sub1_: returned from sub1\n");
  return ret;
}
