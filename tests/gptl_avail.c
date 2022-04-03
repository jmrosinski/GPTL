/** 
 * This is a test of the GPTL. This test will only be run if the PAPI
 * library is present at build-time.
 */

#include "config.h"
#include "gptl.h"
#include <stdio.h>
#include <stdlib.h>

int main ()
{
  int ret;
  char *yes = "Yes";
  char *no  = "No ";
  char *av;                  // Whether derived event is available

  if ((ret = GPTL_PAPIlibraryinit ()) != 0) {
    printf ("Failure from GPTL_PAPIlibraryinit():\n"
	    "Perhaps linking to wrong PAPI version?\n");
    return -1;
  }

  fclose (stderr);  // Disable error messages from GPTL--they can confuse the output here
  
  printf ("Purpose: print availability of PAPI-based derived events on this architecture\n");
  printf ("NOTE: papi_avail from installed PAPI distribution lists available PAPI events\n");
  printf ("      papi_native_avail lists available native events (for advanced users)\n\n");

  printf ("Name                Code Available? Description\n");
  printf ("-----------------------------------------------------------------------\n");

  av = (GPTLsetoption (GPTL_IPC, 1) == 0) ? yes : no;
  printf("%-21s %2d %-10s %s\n", "GPTL_IPC", GPTL_IPC, av, "Instructions per cycle");

  ret = GPTLinitialize ();
  ret = GPTLfinalize ();
  av = (GPTLsetoption (GPTL_LSTPI, 1) == 0) ? yes : no;
  printf("%-21s %2d %-10s %s\n", "GPTL_LSTPI", GPTL_LSTPI, av, "Load-store instruction fraction");

  ret = GPTLinitialize ();
  ret = GPTLfinalize ();
  av = (GPTLsetoption (GPTL_DCMRT, 1) == 0) ? yes : no;
  printf("%-21s %2d %-10s %s\n", "GPTL_DCMRT", GPTL_DCMRT, av, "L1 Miss rate (fraction)");

  ret = GPTLinitialize ();
  ret = GPTLfinalize ();
  av = (GPTLsetoption (GPTL_LSTPDCM, 1) == 0) ? yes : no;
  printf("%-21s %2d %-10s %s\n", "GPTL_LSTPDCM", GPTL_LSTPDCM, av, "Load-store instructions per L1 miss");

  ret = GPTLinitialize ();
  ret = GPTLfinalize ();
  av = (GPTLsetoption (GPTL_L2MRT, 1) == 0) ? yes : no;
  printf("%-21s %2d %-10s %s\n", "GPTL_L2MRT", GPTL_L2MRT, av, "L2 Miss rate (fraction)");

  ret = GPTLinitialize ();
  ret = GPTLfinalize ();
  av = (GPTLsetoption (GPTL_LSTPL2M, 1) == 0) ? yes : no;
  printf("%-21s %2d %-10s %s\n", "GPTL_LSTPL2M", GPTL_LSTPL2M, av, "Load-store instructions per L2 miss");

  ret = GPTLinitialize ();
  ret = GPTLfinalize ();
  av = (GPTLsetoption (GPTL_L3MRT, 1) == 0) ? yes : no;
  printf("%-21s %2d %-10s %s\n", "GPTL_L3MRT", GPTL_L3MRT, av, "L3 Miss rate (fraction)");

  return 0;
}
