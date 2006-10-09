#include <stdio.h>
#include "../gptl.h"

int do_throw (void);
int empty_sub (void);

int main ()
{
  int i;
  int niter;
  int ret;

  GPTLinitialize ();
#ifdef NUMERIC_TIMERS
  GPTLstart (0);
#else
  GPTLstart ("total");
#endif

  printf ("Enter number of iterations\n");
  scanf  ("%d", &niter);

  for (i = 0; i < niter; ++i) {
#ifdef NUMERIC_TIMERS
    GPTLstart (1);
    GPTLstop  (1);
    GPTLstart (2);
#else
    GPTLstart ("do_nothing");
    GPTLstop  ("do_nothing");
    GPTLstart ("empty_sub");
#endif
    ret = empty_sub ();
#ifdef NUMERIC_TIMERS
    GPTLstop  (2);
    GPTLstart (3);
#else
    GPTLstop  ("empty_sub");
    GPTLstart ("exception");
#endif
    try {
      do_throw ();
    }
    catch (...) {
#ifdef NUMERIC_TIMERS
      GPTLstop (3);
#else
      GPTLstop ("exception");
#endif
    }
  }
#ifdef NUMERIC_TIMERS
  GPTLstop (0);
#else
  GPTLstop ("total");
#endif
  GPTLpr (0);
}

int empty_sub (void)
{
  static int i = 7;
  return i;
}

int do_throw (void)
{
  static int i = 7;
  throw (i);
  return i;
}
