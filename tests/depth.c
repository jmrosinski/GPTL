#include <math.h>
#include <sys/time.h>     /* gettimeofday */
#include <unistd.h>       /* gettimeofday */
#include <stdio.h>
#include "../gpt.h"

main()
{
  int niter;
  int i;

  GPTsetoption (GPTcpu, 0);
  GPTsetoption (GPTwall, 1);
  GPTsetoption (GPTabort_on_error, 1);

  GPTinitialize ();

  printf ("Enter number of iterations:\n");
  scanf ("%d", &niter);

  for (i = 0; i < niter; ++i) {
    GPTstart ("muckthingsup2");
    GPTstart ("depth0");
    GPTstart ("muckthingsup");
    GPTstart ("depth1");
    GPTstart ("utilityf");
    GPTstop ("utilityf");
    GPTstart ("depth2");
    GPTstart ("utilityf2");
    GPTstop ("utilityf2");
    GPTstart ("depth3");
    GPTstart ("utilityf");
    GPTstop ("utilityf");
    GPTstart ("depth4");
    GPTstart ("utilityf2");
    GPTstop ("utilityf2");
    GPTstop ("muckthingsup2");
    GPTstop ("depth4");
    GPTstop ("muckthingsup");
    GPTstop ("depth3");
    GPTstop ("depth2");
    GPTstop ("depth1");
    GPTstop ("depth0");
  }

  GPTpr (0);
  GPTfinalize ();
  return 0;
}

