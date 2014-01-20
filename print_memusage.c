/*
** print_memusage.c
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
#include <string.h>
#include <unistd.h>

int GPTLprint_memusage (const char *str)
{
  int size;                       /* process size (returned from OS) */
  int rss;                        /* resident set size (returned from OS) */
  int share;                      /* shared data segment size (returned from OS) */
  int text;                       /* text segment size (returned from OS) */
  int datastack;                  /* data/stack size (returned from OS) */
  static int pagesize = -1;       /* convert to bytes (init to invalid) */
  static double pagestomb = -1;   /* convert pages to MB */
  
  if (GPTLget_memusage (&size, &rss, &share, &text, &datastack) < 0)
    return -1;

#if (defined HAVE_SLASHPROC || defined __APPLE__)

  if (pagesize == -1)
    if ((pagesize = sysconf (_SC_PAGESIZE)) > 0) {
      pagestomb = pagesize / (1024.*1024.);
      printf ("GPTLprint_memusage: Using pagesize=%d\n", pagesize);
    }
  
  if (pagestomb > 0)
    printf ("%s size=%.1f MB rss=%.1f MB datastack=%.1f MB\n", 
	    str, size*pagestomb, rss*pagestomb, datastack*pagestomb);
  else
    printf ("%s size=%d rss=%d datastack=%d\n", 
	    str, size, rss, datastack);

#else

  /*
  ** Use max rss as returned by getrusage. If someone knows how to 
  ** get the process size under AIX please tell me.
  */
  pagesize = 1024;
  pagestomb = pagesize / (1024.*1024.);
  if (1) /* change to 0 if cannot convert to MB */
    printf ("%s max rss=%.1f MB\n", str, rss*pagestomb);
  else
    printf ("%s max rss=%d\n", str, rss);
#endif

  return 0;
}
