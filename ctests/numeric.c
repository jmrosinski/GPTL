#include <stdio.h>
#include "../gptl.h"

#ifdef HAVE_PAPI
#include <papiStdEventDefs.h>
#endif

extern int sub1 (void);
extern int sub2 (void);

int main ()
{
  int i;
  int niter;
  
  printf ("Purpose: test library with NUMERIC_TIMERS enabled\n");

  printf ("Enter number of iterations\n");
  scanf ("%d", &niter);

#ifdef HAVE_PAPI
  GPTLsetoption (PAPI_TOT_CYC, 1);
#endif

  GPTLinitialize ();

  for (i = 0; i < niter; ++i) {
#ifdef NUMERIC_TIMERS
    GPTLstart ((unsigned long) &sub1);
#else
    GPTLstart ("sub1");
#endif
    (void) sub1 ();
#ifdef NUMERIC_TIMERS
    GPTLstop ((unsigned long) &sub1);
    GPTLstart ((unsigned long) &sub2);
#else
    GPTLstop ("sub1");
    GPTLstart ("sub2");
#endif
    (void) sub2 ();
#ifdef NUMERIC_TIMERS
    GPTLstop ((unsigned long) &sub2);

    /* These next 2 should hash to the same value */

    GPTLstart (0xbbbb);
    GPTLstop (0xbbbb);

    GPTLstart (0xbbb);
    GPTLstop (0xbbb);

    GPTLstart (0xabc);
    GPTLstop (0xabc);

    GPTLstart (1);
    GPTLstop (1);

    GPTLstart (2);
    GPTLstop (2);
#else
    GPTLstop ("sub2");

    /* These next 2 should hash to the same value */

    GPTLstart ("bbbb");
    GPTLstop ("bbbb");

    GPTLstart ("bbb");
    GPTLstop ("bbb");

    GPTLstart ("abc");
    GPTLstop ("abc");

    GPTLstart ("1");
    GPTLstop ("1");

    GPTLstart ("2");
    GPTLstop ("2");
#endif
  }

  GPTLpr (0);
  (void) GPTLfinalize ();
}

int sub1 ()
{
  int i;
  static int stuff[1000];

  for (i = 0; i < 1000; i++)
    stuff[i] = i;

  return stuff[999];
}

int sub2 ()
{
  int i;
  static int stuff[1000];

  for (i = 0; i < 1000; i++)
    stuff[i] = i;

  return stuff[999];
}
