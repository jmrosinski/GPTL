//#define _GLIBCXX_CMATH
#include "private.h"

extern "C" {

__host__ static void print_threadmapping (FILE *, int); /* print mapping of thread ids */

__host__ void GPTLprint_memstats (FILE *fp, 
				  Timer **timers, 
				  int nthreads, 
				  int tablesize, 
				  int maxthreads)
{
  Timer *ptr;               /* walk through linked list */
  float pchmem = 0.;        /* parent/child array memory usage */
  float regionmem = 0.;     /* timer memory usage */
  float hashmem;            /* hash table memory usage */
  float callstackmem;       /* callstack memory usage */
  float totmem;             /* total GPTL memory usage */
  int numtimers;            /* number of timers */
  int t;

  hashmem = (float) sizeof (Hashentry) * tablesize * maxthreads;  /* fixed size of table */
  callstackmem = (float) sizeof (Timer *) * MAX_STACK * maxthreads;
  for (t = 0; t < nthreads; t++) {
    numtimers = 0;
    for (ptr = timers[t]->next; ptr; ptr = ptr->next) {
      ++numtimers;
      pchmem  += (float) sizeof (Timer *) * (ptr->nchildren + ptr->nparent);
    }
    hashmem   += (float) numtimers * sizeof (Timer *);
    regionmem += (float) numtimers * sizeof (Timer);
  }

  totmem = hashmem + regionmem + pchmem + callstackmem;
  fprintf (fp, "\n");
  fprintf (fp, "Total GPTL memory usage = %g KB\n", totmem*.001);
  fprintf (fp, "Components:\n");
  fprintf (fp, "Hashmem                 = %g KB\n" 
               "Regionmem               = %g KB\n"
               "Parent/child arrays     = %g KB\n"
               "Callstackmem            = %g KB\n",
           hashmem*.001, regionmem*.001, pchmem*.001, callstackmem*.001);

  print_threadmapping (fp, nthreads);
}

#if ( defined THREADED_OMP )

__host__ static void print_threadmapping (FILE *fp, int nthreads)
{
  int t;

  fprintf (fp, "\n");
  fprintf (fp, "Thread mapping:\n");
  for (t = 0; t < nthreads; ++t)
    fprintf (fp, "GPTLthreadid_omp[%d] = %d\n", t, GPTLthreadid_omp[t]);
}

#else

__host__ static void print_threadmapping (FILE *fp, int nthreads)
{
  fprintf (fp, "\n");
  fprintf (fp, "GPTLthreadid[0] = 0\n");
}

#endif

}
