#include <stdio.h>
#include "../gptl.h"

int main ()
{
  double sum = 0.;
  extern void sub (int, int, char *, double *);

  GPTLinitialize ();

  GPTLstart ("total");
  sub (1, 10000000, "1x1e7", &sum);
  sub (10, 1000000, "10x1e6", &sum);
  sub (100, 100000, "100x1e5", &sum);
  sub (1000, 10000, "1000x1e4", &sum);
  sub (10000, 1000, "1e4x1000", &sum);
  sub (100000, 100, "1e5x100", &sum);
  sub (1000000, 10, "1e6x10", &sum);
  sub (10000000, 1, "1e7x1", &sum);
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





