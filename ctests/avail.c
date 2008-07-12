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

  printf ("Purpose: print PAPI-based events enabled on this architecture\n");
  printf ("Preset events:\n");
  printf ("Name                Code        Description\n");

  i = PAPI_PRESET_MASK;
  do {
    ret = PAPI_get_event_info (i, &info);
    if (ret == PAPI_OK && info.count > 0) {
      printf("%-20s %-10d %s\n", info.symbol, i & info.event_code, info.long_descr);
    }
    /*
    ** Should use PAPI_ENUM_ALL below, but papivi.h doesn't always
    ** automatically get installed
    */
  } while (PAPI_enum_event (&i, 0) == PAPI_OK);
  /*
  } while (PAPI_enum_event (&i, PAPI_ENUM_ALL) == PAPI_OK);
  */

  printf ("\n\n\n");
  printf ("Native events:\n");
  printf ("Name                Code        Description\n");
  i = PAPI_NATIVE_MASK;
  do {
    ret = PAPI_get_event_info (i, &info);
    if (ret == PAPI_OK && info.count > 0) {
      printf("%-20s %-10d %s\n", info.symbol, i & info.event_code, info.long_descr);
    }
    /*
    ** Should use PAPI_ENUM_ALL below, but papivi.h doesn't always
    ** automatically get installed
    */
  } while (PAPI_enum_event (&i, 0) == PAPI_OK);

  printf ("\n\n\n");
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
