#include <math.h>
#include <sys/time.h>     /* gettimeofday */
#include <unistd.h>       /* gettimeofday */
#include <stdio.h>
#include "../gptl.h"

int main(int argc, char **argv)
{
  int niter;
  int i;

#ifdef NUMERIC_TIMERS
  printf ("%s not enabled for NUMERIC_TIMERS\n", argv[0]);
  exit (-1);
#else
  GPTLsetoption (GPTLcpu, 0);
  GPTLsetoption (GPTLwall, 1);
  GPTLsetoption (GPTLabort_on_error, 1);

  GPTLinitialize ();

  printf ("Enter number of iterations:\n");
  scanf ("%d", &niter);

  for (i = 0; i < niter; ++i) {
    GPTLstart ("muckthingsup2");
    GPTLstart ("depth0");
    GPTLstart ("muckthingsup");
    GPTLstart ("depth1");
    GPTLstart ("utilityf");
    GPTLstop ("utilityf");
    GPTLstart ("depth2");
    GPTLstart ("utilityf2");
    GPTLstop ("utilityf2");
    GPTLstart ("depth3");
    GPTLstart ("utilityf");
    GPTLstop ("utilityf");
    GPTLstart ("depth4");
    GPTLstart ("utilityf2");
    GPTLstop ("utilityf2");
    GPTLstop ("muckthingsup2");
    GPTLstop ("depth4");
    GPTLstop ("muckthingsup");
    GPTLstop ("depth3");
    GPTLstop ("depth2");
    GPTLstop ("depth1");
    GPTLstop ("depth0");
  }

  GPTLpr (0);
  GPTLfinalize ();
#endif
  return 0;
}
