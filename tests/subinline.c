#include <stdio.h>
#include "../gptl.h"

static inline void subinline (int);
void subnotinline (int);

int main (int argc, char **argv)
{
  int ncalls;
  int i;

  printf ("Determine whether a function got inlined.  Designed to be called "
	  "with -DNUMERIC_TIMERS\n");

  printf ("Enter number of calls\n");
  scanf ("%d", &ncalls);

  GPTLinitialize ();
#ifndef NUMERIC_TIMERS
  GPTLstart ("total");
#endif
  for (i = 0; i < ncalls; i++) {
    subinline (i);
    subnotinline (i);
  }
#ifndef NUMERIC_TIMERS
  GPTLstop ("total");
#endif

  GPTLpr (0);
  (void) GPTLfinalize ();
}

static inline void subinline (int i)
{
  int j;
  float sum = 0;
  
#ifndef NUMERIC_TIMERS
  GPTLstart ("subinline");
#endif
  for (j = 0; j < i; j++)
    sum += j;
  printf ("sum=%f\n", sum);
#ifndef NUMERIC_TIMERS
  GPTLstop ("subinline");
#endif
  return;
}

void subnotinline (int i)
{
  int j;
  float sum = 0;

#ifndef NUMERIC_TIMERS
  GPTLstart ("subnotinline");
#endif
  for (j = 0; j < i; j++)
    sum += j;
  printf ("sum=%f\n", sum);
#ifndef NUMERIC_TIMERS
  GPTLstop ("subnotinline");
#endif
  return;
}
