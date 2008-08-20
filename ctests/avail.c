#include <stdio.h>
#include <stdlib.h>
#include "../private.h"

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
  printf ("Derived events:\n");
  printf ("Name                Code        Description\n");

  /*
  ** This should be done with a "getter" function for the derived counter name
  */

  if ((ret == PAPI_query_event (PAPI_TOT_INS)) == PAPI_OK &&
      (ret == PAPI_query_event (PAPI_TOT_CYC)) == PAPI_OK)
    printf("%-20s %-10d %s\n", "GPTL_IPC", GPTL_IPC, "Instructions per cycle\n");

  if ((ret == PAPI_query_event (PAPI_FP_OPS)) == PAPI_OK &&
      (ret == PAPI_query_event (PAPI_LST_INS)) == PAPI_OK)
    printf("%-20s %-10d %s\n", "GPTL_IPC", GPTL_CI, "Computational intensity\n");
  
#else
  printf ("PAPI not enabled so this code does nothing\n");
#endif
  return 0;
}
