#include <stdio.h>
#include "../gptl.h"

#ifdef HAVE_PAPI
#include <papiStdEventDefs.h>
#endif

int main (int argc, char **argv)
{
  char timername[16];
  int i;
  int ntimers;

#ifdef NUMERIC_TIMERS
  printf ("%s not enabled for NUMERIC_TIMERS\n", argv[0]);
  return (-1);
#else
  printf ("Purpose: compare timings of basetimer1, basetimer2 and basetimer3.\n"
	  "Difference is measure of cost of traversing linked list\n");

  printf ("Enter number of intermediate timers\n");
  scanf ("%d", &ntimers);

#ifdef HAVE_PAPI
  GPTLsetoption (PAPI_TOT_CYC, 1);
#endif

  GPTLinitialize ();
  GPTLstart ("basetimer1");
  GPTLstop ("basetimer1");
  for (i = 0; i < ntimers/2; ++i) {
    sprintf (timername, "midtimer%4.4d", i);
    GPTLstart (timername);
    GPTLstop (timername);
  }

  GPTLstart ("basetimer2");
  GPTLstop ("basetimer2");

  for (i = ntimers/2+1; i < ntimers; ++i) {
    sprintf (timername, "midtimer%4.4d", i);
    GPTLstart (timername);
    GPTLstop (timername);
  }

  for (i = ntimers/2+1; i < ntimers; ++i) {
    sprintf (timername, "midtimer%4.4d", i);
    GPTLstart (timername);
    GPTLstop (timername);
  }

  GPTLstart ("basetimer3");
  GPTLstop ("basetimer3");

  for (i = 0; i < 1000; ++i) {
    GPTLstart ("basetimer1");
    GPTLstop ("basetimer1");
    
    GPTLstart ("basetimer2");
    GPTLstop ("basetimer2");

    GPTLstart ("basetimer3");
    GPTLstop ("basetimer3");
  }

  GPTLpr (0);
  (void) GPTLfinalize ();
#endif
}
