#include <stdio.h>
#include <stdlib.h>
#include "../gptl.h"

#ifdef HAVE_PAPI
#include <papi.h>
#endif

int main ()
{
#ifdef HAVE_PAPI

  int i;
  int ret;
  PAPI_event_info_t info;

  if ((ret = PAPI_library_init (PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) {
    printf (PAPI_strerror (ret));
    return -1;
  }

  printf ("Purpose: print derived events available on this architecture\n");
  printf ("For PAPI-specific events, run papi_avail and papi_native_avail"
	  " from the PAPI distribution\n\n");
  printf ("Available derived events:\n");
  printf ("Name                Code        Description\n");
  printf ("-------------------------------------------\n");

  /*
  ** This should be done with a "getter" function for the derived counter name
  */

  if (GPTLsetoption (GPTL_IPC, 1) == 0)
    printf("%-20s %-10d %s\n", "GPTL_IPC", GPTL_IPC, "Instructions per cycle");

  if (GPTLsetoption (GPTL_CI, 1) == 0)
    printf("%-20s %-10d %s\n", "GPTL_CI", GPTL_CI, "Computational intensity");

#else
  printf ("PAPI not enabled so this code does nothing\n");
#endif
  return 0;
}
