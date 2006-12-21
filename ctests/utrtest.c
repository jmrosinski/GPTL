#include <stdio.h>
#include "../gptl.h"

int main ()
{
  double sum;
  extern void sub (int, int, char *, double *);

  GPTLsetutr (GPTLnanotime);
  GPTLsetutr (GPTLrtc);
  GPTLsetutr (GPTLmpiwtime);
  GPTLsetutr (GPTLclockgettime);
  //  GPTLsetutr (GPTLgettimeofday);

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
