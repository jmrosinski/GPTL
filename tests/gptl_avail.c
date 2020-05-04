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

  if ((ret = GPTL_PAPIlibraryinit ()) != 0) {
    printf ("Failure from GPTL_PAPIlibraryinit():\n"
	    "Perhaps linking to wrong PAPI version?\n");
    return -1;
  }

  printf ("Purpose: print derived events available on this architecture\n");
  printf ("Note: 'available' may require enabling multiplexing in some cases\n");
  printf ("For PAPI-specific events, run papi_avail and papi_native_avail"
	  " from the PAPI distribution\n\n");
  printf ("Available derived events:\n");
  printf ("Name                Code        Description\n");
  printf ("-------------------------------------------\n");

  if (GPTLsetoption (GPTL_IPC, 1) == 0) {
    printf("%-20s %-10d %s\n", "GPTL_IPC", GPTL_IPC, "Instructions per cycle");
    ret = GPTLinitialize ();
    ret = GPTLfinalize ();
  }

  if (GPTLsetoption (GPTL_LSTPI, 1) == 0) {
    printf("%-20s %-10d %s\n", "GPTL_LSTPI", GPTL_LSTPI, "Load-store instruction fraction");
    ret = GPTLinitialize ();
    ret = GPTLfinalize ();
  }

  if (GPTLsetoption (GPTL_DCMRT, 1) == 0) {
    printf("%-20s %-10d %s\n", "GPTL_DCMRT", GPTL_DCMRT, "L1 Miss rate (fraction)");
    ret = GPTLinitialize ();
    ret = GPTLfinalize ();
  }

  if (GPTLsetoption (GPTL_LSTPDCM, 1) == 0) {
    printf("%-20s %-10d %s\n", "GPTL_LSTPDCM", GPTL_LSTPDCM, "Load-store instructions per L1 miss");
    ret = GPTLinitialize ();
    ret = GPTLfinalize ();
  }

  if (GPTLsetoption (GPTL_L2MRT, 1) == 0) {
    printf("%-20s %-10d %s\n", "GPTL_L2MRT", GPTL_L2MRT, "L2 Miss rate (fraction)");
    ret = GPTLinitialize ();
    ret = GPTLfinalize ();
  }

  if (GPTLsetoption (GPTL_LSTPL2M, 1) == 0) {
    printf("%-20s %-10d %s\n", "GPTL_LSTPL2M", GPTL_LSTPL2M, "Load-store instructions per L2 miss");
    ret = GPTLinitialize ();
    ret = GPTLfinalize ();
  }
  if (GPTLsetoption (GPTL_L3MRT, 1) == 0) {
    printf("%-20s %-10d %s\n", "GPTL_L3MRT", GPTL_L3MRT, "L3 Miss rate (fraction)");
    ret = GPTLinitialize ();
    ret = GPTLfinalize ();
  }

  return 0;
}
