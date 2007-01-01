#include <stdio.h>
#include "../gptl.h"

#ifdef NUMERIC_TIMERS
#define TOTAL 0
#define SUB 1
#define LOOP 2
#else
#define TOTAL "total"
#define SUB "sub"
#define LOOP "loop"
#endif

#ifdef HAVE_PAPI
#include <papiStdEventDefs.h>
#endif

void sub (int);
int main (int argc, char **argv)
{
  int depth;

  printf ("Purpose: compare timings of basetimer1, basetimer2 and basetimer3.\n"
	  "Difference is measure of cost of traversing linked list\n");

  printf ("Enter recursion depth\n");
  scanf ("%d", &depth);

#ifdef HAVE_PAPI
  GPTLsetoption (PAPI_TOT_CYC, 1);
#endif

  GPTLinitialize ();
  GPTLstart (TOTAL);
  sub (depth);
  GPTLstop (TOTAL);

  GPTLpr (0);
  (void) GPTLfinalize ();
  return 0;
}

void sub (int depth)
{
  int i;
  float sum = 0;

  GPTLstart (SUB);
  GPTLstart (LOOP);

  for (i = 0; i < depth; i++)
    sum += depth*(1. + sum);

  GPTLstop (LOOP);

  printf ("sum=%f\n", sum);
  if (depth <= 0) {
    GPTLstop (SUB);
    return;
  }
  sub (depth - 1);
  GPTLstop (SUB);
  return;
}
