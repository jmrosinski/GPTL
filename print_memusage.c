/*
** print_memusage:
**
**   Prints info about memory usage of this process by calling get_memusage.
**
**   Return value: 0  = success
**                 -1 = failure
*/

#include <stdio.h>
#include "gptl.h"

#if ( defined FORTRANCAPS )
#define print_memusage PRINT_MEMUSAGE
#elif ( defined FORTRANUNDERSCORE )
#define print_memusage print_memusage_
#elif ( defined FORTRANDOUBLEUNDERSCORE )
#define print_memusage print_memusage__
#endif

int print_memusage (char *str)
{
  int size;
  int rss;
  int share;
  int text;
  int datastack;

  if (get_memusage (&size, &rss, &share, &text, &datastack) < 0)
    return -1;

  printf ("%s size=%d rss=%d share=%d text=%d datastack=%d\n", 
	  str, size, rss, share, text, datastack);
  return 0;
}
