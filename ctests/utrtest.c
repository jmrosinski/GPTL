#include <stdio.h>
#include "../gptl.h"
#include <mpi.h>

int main (int argc, char **argv)
{
  double sum;
  extern void sub (int, int, char *, double *);

  MPI_Init (&argc, &argv);
  GPTLsetutr (GPTLrtc);
  GPTLsetutr (GPTLnanotime);
  GPTLsetutr (GPTLclockgettime);
  GPTLsetutr (GPTLmpiwtime);
  GPTLsetutr (GPTLgettimeofday);

  GPTLinitialize ();

  GPTLstart ("total");
  sub (1000000, 1, "1e6x1", &sum);
  GPTLstop ("total");

  GPTLpr (0);
  return 0;
}

void sub (int outer, int inner, char *name, double *sum)
{
  int i, j;
  for (i = 0; i < outer; ++i) {
    GPTLstart (name);
    for (j = 0; j < inner; ++j)
      *sum += j;
    GPTLstop (name);
  }
}
