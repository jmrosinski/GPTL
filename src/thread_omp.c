/*
**
** Author: Jim Rosinski
** 
** Utility functions for OpenMP-based threading
*/

#include "config.h"   // Must be first include
#include "thread.h"
#include "private.h"
#include "gptl_papi.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>   // free

volatile int GPTLnthreads = -1;        // num threads: init to bad value
volatile int GPTLmax_threads = -1;     // max num threads
static volatile int *threadid = NULL;  // array of thread ids

/*
** GPTLthreadinit: Allocate and initialize threadid; set max number of threads
**
** Output results:
**   GPTLmax_threads: max number of threads
**
**   threadid[] is allocated and initialized to -1
**
**
** Return value: 0 (success) or GPTLerror (failure)
*/
int GPTLthreadinit (void)
{
  int t;
  static const char *thisfunc = "GPTLthreadinit";

  if (omp_get_thread_num () != 0)
    return GPTLerror ("OMP %s: MUST only be called by the master thread\n", thisfunc);

  // Allocate the threadid array which maps physical thread IDs to logical IDs 
  // For OpenMP this will be just threadid[iam] = iam;
  if (threadid) 
    return GPTLerror ("OMP %s: has already been called.\n"
		      "Maybe mistakenly called by multiple threads?\n", thisfunc);

  // GPTLmax_threads may have been set by the user, in which case use that. But if as 
  // yet uninitialized, set to the current value of OMP_NUM_THREADS. 
  if (GPTLmax_threads == -1)
    GPTLmax_threads = MAX ((1), (omp_get_max_threads ()));

  if ( ! (threadid = (int *) GPTLallocate (GPTLmax_threads * sizeof (int), thisfunc)))
    return GPTLerror ("OMP %s: malloc failure for %d elements of threadid\n",
		      thisfunc, GPTLmax_threads);

  // Initialize threadid array to flag values for use by GPTLget_thread_num().
  // get_thread_num() will fill in the values on first use.
  for (t = 0; t < GPTLmax_threads; ++t)
    threadid[t] = -1;
#ifdef VERBOSE
  printf ("GPTL: OMP %s: Set GPTLmax_threads=%d\n", thisfunc, GPTLmax_threads);
#endif
  return 0;
}

/*
** GPTLthreadfinalize: clean up
**
** Output results:
**   threadid array is freed and array pointer nullified
*/
void GPTLthreadfinalize ()
{
  free ((void *) threadid);
  threadid = 0;
}

/*
** GPTLget_thread_num: Determine thread number of the calling thread
**                     Start PAPI counters if enabled and first call for this thread.
**
** Output results:
**   GPTLnthreads:     Number of threads
**   threadid: Our thread id added to list on 1st call
**
** Return value: thread number (success) or GPTLerror (failure)
**   5/8/16: Modified to enable 2-level OMP nesting: Fold combination of current and parent
**   thread info into a single index
*/
int GPTLget_thread_num (void)
{
  int t;
  static const char *thisfunc = "GPTLget_thread_num";

#ifdef ENABLE_NESTEDOMP
  int myid;            // my thread id
  int lvl;             // nest level: Currently only 2 nesting levels supported
  int parentid;        // thread number of parent team
  int my_nthreads;     // number of threads in the parent team

  myid = omp_get_thread_num ();
  if (omp_get_nested ()) {         // nesting is "enabled", though not necessarily active
    lvl = omp_get_active_level (); // lvl=2 => inside 2 #pragma omp regions
    if (lvl < 2) {
      // 0 or 1-level deep: simply use thread id as index
      t = myid;
    } else if (lvl == 2) {
      // Create a unique id "t" for indexing into singly-dimensioned thread array
      parentid    = omp_get_ancestor_thread_num (lvl-1);
      my_nthreads = omp_get_team_size (lvl);
      t           = parentid*my_nthreads + myid;
    } else {
      return GPTLerror ("OMP %s: GPTL supports only 2 nested OMP levels got %d\n", thisfunc, lvl);
    }
  } else {
    // un-nested case: thread id is index
    t = myid;
  }
#else
  t = omp_get_thread_num ();
#endif
  if (t >= GPTLmax_threads)
    return GPTLerror ("OMP %s: returned id=%d exceeds GPTLmax_threads=%d\n",
		      thisfunc, t, GPTLmax_threads);

  // If our thread number has already been set in the list, we are done
  if (t == threadid[t])
    return t;

  // Thread id not found. Modify threadid with our ID, then start PAPI events if required.
  // Due to the setting of threadid, everything below here will only execute once per thread.
  threadid[t] = t;

#ifdef VERBOSE
  printf ("GPTL: OMP %s: 1st call t=%d\n", thisfunc, t);
#endif

#ifdef HAVE_PAPI

  // When HAVE_PAPI is true, if 1 or more PAPI events are enabled,
  // create and start an event set for the new thread.
  if (GPTLget_npapievents () > 0) {
#ifdef VERBOSE
    printf ("GPTL: OMP %s: Starting EventSet t=%d\n", thisfunc, t);
#endif

    if (GPTLcreate_and_start_events (t) < 0)
      return GPTLerror ("GPTL: OMP %s: error from GPTLcreate_and_start_events for thread %d\n", 
			thisfunc, t);
  }
#endif

  // nthreads = GPTLmax_threads based on setting in GPTLthreadinit or user call to GPTLsetoption()
  GPTLnthreads = GPTLmax_threads;
#ifdef VERBOSE
  printf ("GPTL: OMP %s: nthreads=%d\n", thisfunc, GPTLnthreads);
#endif

  return t;
}

void GPTLprint_threadmapping (FILE *fp)
{
  int t;
  fprintf (fp, "\n");
  fprintf (fp, "Thread mapping:\n");
  for (t = 0; t < GPTLnthreads; ++t)
    fprintf (fp, "threadid[%d] = %d\n", t, threadid[t]);
}
