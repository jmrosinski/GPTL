#include <stdio.h>
#include "../gpt.h"

#ifdef HAVE_PAPI
#include <papiStdEventDefs.h>
#endif

int main ()
{
  char timername[16];
  int i;
  int ntimers;
  
  printf ("Purpose: compare timings of basetimer1, basetimer2 and basetimer3.\n"
	  "Difference is measure of cost of traversing linked list\n");

  printf ("Enter number of intermediate timers\n");
  scanf ("%d", &ntimers);

#ifdef HAVE_PAPI
  GPTsetoption (PAPI_TOT_CYC, 1);
#endif

  GPTinitialize ();
  GPTstart ("basetimer1");
  GPTstop ("basetimer1");
  for (i = 0; i < ntimers/2; ++i) {
    sprintf (timername, "midtimer%4.4d", i);
    GPTstart (timername);
    GPTstop (timername);
  }

  GPTstart ("basetimer2");
  GPTstop ("basetimer2");

  for (i = ntimers/2+1; i < ntimers; ++i) {
    sprintf (timername, "midtimer%4.4d", i);
    GPTstart (timername);
    GPTstop (timername);
  }

  for (i = ntimers/2+1; i < ntimers; ++i) {
    sprintf (timername, "midtimer%4.4d", i);
    GPTstart (timername);
    GPTstop (timername);
  }

  GPTstart ("basetimer3");
  GPTstop ("basetimer3");

  for (i = 0; i < 1000; ++i) {
    GPTstart ("basetimer1");
    GPTstop ("basetimer1");
    
    GPTstart ("basetimer2");
    GPTstop ("basetimer2");

    GPTstart ("basetimer3");
    GPTstop ("basetimer3");
  }

  GPTpr (0);
  (void) GPTfinalize ();
}
