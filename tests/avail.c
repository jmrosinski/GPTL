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
  
  printf("Name                Code        Description\n");
  i = PAPI_PRESET_MASK;

  do {
    ret = PAPI_get_event_info (i, &info);
    if (ret == PAPI_OK && info.count > 0) {
      printf("%-20s %-10d %s\n", info.symbol, i & info.event_code, info.long_descr);
    }
  } while (PAPI_enum_event (&i, PAPI_ENUM_ALL) == PAPI_OK);
#else
  printf ("PAPI not enabled so this code does nothing\n");
#endif
}
