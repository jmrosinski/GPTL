/*
** $Id: print_memusage.c,v 1.6 2010-03-26 05:02:23 rosinski Exp $
**
** Author: Jim Rosinski
**
** print_memusage:
**
**   Prints info about memory usage of this process by calling get_memusage.
**
**   Return value: 0  = success
**                 -1 = failure
*/

#include "gptl.h"
#include <stdio.h>
#include <stdlib.h>

int GPTLprint_memusage (const char *str)
{
  int size, size2;
  int rss, rss2;
  int share, share2;
  int text, text2;
  int datastack, datastack2;
  static int bytesperblock = -1;          /* convert to bytes (init to invalid) */
  static const int nbytes = 1024*1024*10; /* allocate 10 MB */
  void *space;                            /* allocated space */
  double blockstomb;                      /* convert blocks to MB */
  
  if (GPTLget_memusage (&size, &rss, &share, &text, &datastack) < 0)
    return -1;

  if (bytesperblock == -1 && (space = malloc (nbytes))) {
    if (GPTLget_memusage (&size2, &rss2, &share2, &text2, &datastack2) == 0) {
      bytesperblock = (int) ((nbytes / (double) (size2 - size)) + 0.5);
      blockstomb = bytesperblock / (1024.*1024.);
      printf ("GPTLprint_memusage: Using bytesperblock=%d\n", bytesperblock);
    }
    free (space);
  }
  
  if (bytesperblock > 0)
    printf ("%s size=%f.1 MB rss=%f.1 MB share=%f.1 MB text=%f.1 MB datastack=%f.1 MB\n", 
	    str, size*blockstomb, rss*blockstomb, share*blockstomb, 
	    text*blockstomb, datastack*blockstomb);
  else
    printf ("%s size=%d rss=%d share=%d text=%d datastack=%d\n", 
	    str, size, rss, share, text, datastack);

  return 0;
}
