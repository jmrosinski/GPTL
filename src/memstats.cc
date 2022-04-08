/*
** memstats.c
**
** Author: Jim Rosinski
**
** Print stats about GPTL memory usage
*/
 
#include "config.h"  // Must be first include
#include "private.h"
#include "thread.h"

void GPTLprint_memstats (FILE *fp, Timer **timers, int tablesize)
{
  Timer *ptr;               // walk through linked list
  float pchmem = 0.;        // parent/child array memory usage
  float regionmem = 0.;     // timer memory usage
  float papimem = 0.;       // PAPI stats memory usage
  float hashmem;            // hash table memory usage
  float callstackmem;       // callstack memory usage
  float totmem;             // total GPTL memory usage
  int numtimers;            // number of timers
  int t;

  hashmem = (float) sizeof (Hashentry) * tablesize * GPTLmax_threads;  // fixed size of table
  callstackmem = (float) sizeof (Timer *) * MAX_STACK * GPTLmax_threads;
  for (t = 0; t < GPTLnthreads; t++) {
    numtimers = 0;
    for (ptr = timers[t]->next; ptr; ptr = ptr->next) {
      ++numtimers;
      pchmem  += (float) sizeof (Timer *) * (ptr->nchildren + ptr->nparent);
    }
    hashmem   += (float) numtimers * sizeof (Timer *);
    regionmem += (float) numtimers * sizeof (Timer);
#ifdef HAVE_PAPI
    papimem += (float) numtimers * sizeof (Papistats);
#endif
  }

  totmem = hashmem + regionmem + pchmem + callstackmem;
  fprintf (fp, "\n");
  fprintf (fp, "Total GPTL memory usage = %g KB\n", totmem*.001);
  fprintf (fp, "Components:\n");
  fprintf (fp, "Hashmem                 = %g KB\n" 
               "Regionmem               = %g KB (papimem portion = %g KB)\n"
               "Parent/child arrays     = %g KB\n"
               "Callstackmem            = %g KB\n",
           hashmem*.001, regionmem*.001, papimem*.001, pchmem*.001, callstackmem*.001);

  GPTLprint_threadmapping (fp);
}
