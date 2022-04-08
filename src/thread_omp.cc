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

namespace thread {
  volatile int max_threads = -1;     // max num threads;
  volatile int nthreads = -1;        // num threads: init to bad value;
  volatile int *threadid = NULL;     // array of thread ids
/*
** threadinit: Allocate and initialize threadid; set max number of threads
**
** Output results:
**   max_threads: max number of threads
**
**   threadid[] is allocated and initialized to -1
**
**
** Return value: 0 (success) or GPTLerror (failure)
*/
  int threadinit (void)
  {
    int t;
    static const char *thisfunc = "threadinit";

    if (omp_get_thread_num () != 0)
      return GPTLerror ("OMP %s: MUST only be called by the master thread\n", thisfunc);

    // Allocate the threadid array which maps physical thread IDs to logical IDs 
    // For OpenMP this will be just threadid[iam] = iam;
    if (threadid) 
      return GPTLerror ("OMP %s: has already been called.\n"
			"Maybe mistakenly called by multiple threads?\n", thisfunc);

    // max_threads may have been set by the user, in which case use that. But if as 
    // yet uninitialized, set to the current value of OMP_NUM_THREADS or a user call to
    // omp_set_num_threads ()
    if (max_threads == -1)
      max_threads = MAX ((1), (omp_get_max_threads ()));
    
    if ( ! (threadid = (int *) GPTLallocate (max_threads * sizeof (int), thisfunc)))
      return GPTLerror ("OMP %s: malloc failure for %d elements of threadid\n",
			thisfunc, max_threads);

    // Initialize threadid array to flag values for use by get_thread_num().
    // get_thread_num() will fill in the values on first use.
    for (t = 0; t < max_threads; ++t)
      threadid[t] = -1;
#ifdef VERBOSE
    printf ("GPTL: OMP %s: Set max_threads=%d\n", thisfunc, max_threads);
#endif
    return 0;
  }

  /*
  ** threadfinalize: clean up
  **
  ** Output results:
  **   threadid array is freed and array pointer nullified
  */
  void threadfinalize ()
  {
    free ((void *) threadid);
    threadid = NULL;
  }

  /*
  ** get_thread_num: Determine thread number of the calling thread
  **                     Start PAPI counters if enabled and first call for this thread.
  **
  ** Output results:
  **   nthreads:     Number of threads
  **   threadid: Our thread id added to list on 1st call
  **
  ** Return value: thread number (success) or GPTLerror (failure)
  **   5/8/16: Modified to enable 2-level OMP nesting: Fold combination of current and parent
  **   thread info into a single index
  */
#ifdef INLINE_THREADING
  inline
#endif
  int get_thread_num (void)
  {
    int t = omp_get_thread_num ();     // linearized thread id to be returned
    static const char *thisfunc = "get_thread_num";

#if ( defined ENABLE_NESTEDOMP )
    const int myid = t;              // omp_get_thread_num ();
    if (omp_get_max_active_levels () > 1) { // nesting is "enabled", though not necessarily active
      const int lvl = omp_get_active_level (); // lvl=2 => inside 2 #pragma omp regions
      // Support for OpenMP 5.2 standard required for more than 2 nested OMP levels
#ifdef LESSTHAN_OMP52
      if (lvl == 2) {
	if (myid == 0) {
	  t = omp_get_ancestor_thread_num (1);   // thread 0 gets parent id
	  // First -1 is to increment from omp_get_team_size(1)-1
	  // Second -1 is because thread 0 of the team took parent's id
	} else {
	  t = omp_get_team_size (1) - 1 +
	    omp_get_ancestor_thread_num (1)*(omp_get_team_size (2) - 1) + myid;
	}
      } else if (lvl > 2) {
	return GPTLerror ("OMP %s: GPTL supports only 2 nested OMP levels got %d\n", thisfunc, lvl);
      }
#else
      // THIS BLOCK OF CODE NEEDS TESTING ONCE OMP_GET_TEAM_NUM() SUCCEEDS
      if (lvl > 1) {
	// Create a unique id "t" for indexing into singly-dimensioned thread array
	int team_size   = omp_get_team_size (1);    // 1
	int num_teams   = 1;
	int tot_threads = team_size;
	for (int d = 2; d < lvl; ++d) {
	  team_size    = omp_get_team_size (d);  // curr
	  num_teams   *= team_size;             // curr
	  tot_threads += num_teams*team_size; // curr sum
	}
	t = tot_threads + omp_get_team_num ()*omp_get_team_size (lvl) + myid;
	printf ("lvl %d tt %d team_num %d team_size %d myid %d t %d\n", lvl, tot_threads,
		omp_get_team_num(), omp_get_team_size(lvl), myid, t);
	printf ("lvl %d num_teams %d\n", lvl, omp_get_num_teams());
      }
#endif
    }
#endif
    if (t >= max_threads)
      return GPTLerror ("OMP %s: returned id=%d exceeds max_threads=%d\n",
			thisfunc, t, max_threads);

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
    if (get_npapievents () > 0) {
#ifdef VERBOSE
      printf ("GPTL: OMP %s: Starting EventSet t=%d\n", thisfunc, t);
#endif
      if (create_and_start_events (t) < 0)
	return GPTLerror ("GPTL: OMP %s: error from create_and_start_events for thread %d\n", 
			  thisfunc, t);
    }
#endif

    // nthreads = max_threads based on setting in threadinit or user call to setoption()
    nthreads = max_threads;
#ifdef VERBOSE
    printf ("GPTL: OMP %s: nthreads=%d\n", thisfunc, nthreads);
#endif

    return t;
  }

  void print_threadmapping (FILE *fp)
  {
    int t;
    fprintf (fp, "\n");
    fprintf (fp, "Thread mapping:\n");
    for (t = 0; t < nthreads; ++t)
      fprintf (fp, "threadid[%d] = %d\n", t, threadid[t]);
  }
}
