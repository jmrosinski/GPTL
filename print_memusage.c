/*
** $Id: print_memusage.c,v 1.7 2010-03-27 18:39:00 rosinski Exp $
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

static int nearest_powerof2 (int);
static int convert_to_bytes = 1;   /* true */

int GPTLprint_memusage (const char *str)
{
  int size, size2;
  int rss, rss2;
  int share, share2;
  int text, text2;
  int datastack, datastack2;
  static int bytesperblock = -1;          /* convert to bytes (init to invalid) */
  static const int nbytes = 1024*1024*10; /* allocate 10 MB */
  static double blockstomb;                      /* convert blocks to MB */
  void *space;                            /* allocated space */
  
  if (GPTLget_memusage (&size, &rss, &share, &text, &datastack) < 0)
    return -1;

  if (convert_to_bytes && bytesperblock == -1 && (space = malloc (nbytes))) {
    if (GPTLget_memusage (&size2, &rss2, &share2, &text2, &datastack2) == 0) {
      /*
      ** Estimate bytes per block, then refine to nearest power of 2
      */
      bytesperblock = (int) ((nbytes / (double) (size2 - size)) + 0.5);
      bytesperblock = nearest_powerof2 (bytesperblock);
      blockstomb = bytesperblock / (1024.*1024.);
      printf ("GPTLprint_memusage: Using bytesperblock=%d\n", bytesperblock);
    }
    free (space);
  }
  
  if (bytesperblock > 0)
    printf ("%s size=%.1f MB rss=%.1f MB share=%.1f MB text=%.1f MB datastack=%.1f MB\n", 
	    str, size*blockstomb, rss*blockstomb, share*blockstomb, 
	    text*blockstomb, datastack*blockstomb);
  else
    printf ("%s size=%d rss=%d share=%d text=%d datastack=%d\n", 
	    str, size, rss, share, text, datastack);

  return 0;
}

static int nearest_powerof2 (int bytesperblock)
{
  int lower;
  int higher;
  int delta1;
  int delta2;

  if (bytesperblock < 2)
    return 0;

  for (higher = 1; higher < bytesperblock; higher *= 2)
    lower = higher;

  delta1 = bytesperblock - lower;
  delta2 = higher - bytesperblock;
  
  if (delta1 < delta2)
    return lower;
  else
    return higher;
}
