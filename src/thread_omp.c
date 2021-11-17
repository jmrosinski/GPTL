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
// make threadid non-static due to this file possibly being inlined
volatile int *GPTLthreadid = NULL;  // array of thread ids

/*
** GPTLthreadinit: Allocate and initialize GPTLthreadid; set max number of threads
**
** Output results:
**   GPTLmax_threads: max number of threads
**
**   GPTLthreadid[] is allocated and initialized to -1
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

  // Allocate the GPTLthreadid array which maps physical thread IDs to logical IDs 
  // For OpenMP this will be just GPTLthreadid[iam] = iam;
  if (GPTLthreadid) 
    return GPTLerror ("OMP %s: has already been called.\n"
		      "Maybe mistakenly called by multiple threads?\n", thisfunc);

  // GPTLmax_threads may have been set by the user, in which case use that. But if as 
  // yet uninitialized, set to the current value of OMP_NUM_THREADS or a user call to
  // omp_set_num_threads ()
  if (GPTLmax_threads == -1)
    GPTLmax_threads = MAX ((1), (omp_get_max_threads ()));

  if ( ! (GPTLthreadid = (int *) GPTLallocate (GPTLmax_threads * sizeof (int), thisfunc)))
    return GPTLerror ("OMP %s: malloc failure for %d elements of GPTLthreadid\n",
		      thisfunc, GPTLmax_threads);

  // Initialize GPTLthreadid array to flag values for use by GPTLget_thread_num().
  // get_thread_num() will fill in the values on first use.
  for (t = 0; t < GPTLmax_threads; ++t)
    GPTLthreadid[t] = -1;
#ifdef VERBOSE
  printf ("GPTL: OMP %s: Set GPTLmax_threads=%d\n", thisfunc, GPTLmax_threads);
#endif
  return 0;
}

/*
** GPTLthreadfinalize: clean up
**
** Output results:
**   GPTLthreadid array is freed and array pointer nullified
*/
void GPTLthreadfinalize ()
{
  free ((void *) GPTLthreadid);
  GPTLthreadid = NULL;
}

/*
** GPTLget_thread_num: Determine thread number of the calling thread
**                     Start PAPI counters if enabled and first call for this thread.
**
** Output results:
**   GPTLnthreads:     Number of threads
**   GPTLthreadid: Our thread id added to list on 1st call
**
** Return value: thread number (success) or GPTLerror (failure)
**   5/8/16: Modified to enable 2-level OMP nesting: Fold combination of current and parent
**   thread info into a single index
*/
#ifdef INLINE_THREADING
inline
#endif
int GPTLget_thread_num (void)
{
  int t = omp_get_thread_num ();     // linearized thread id to be returned
  static const char *thisfunc = "GPTLget_thread_num";

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
  if (t >= GPTLmax_threads)
    return GPTLerror ("OMP %s: returned id=%d exceeds GPTLmax_threads=%d\n",
		      thisfunc, t, GPTLmax_threads);

  // If our thread number has already been set in the list, we are done
  if (t == GPTLthreadid[t])
    return t;

  // Thread id not found. Modify GPTLthreadid with our ID, then start PAPI events if required.
  // Due to the setting of GPTLthreadid, everything below here will only execute once per thread.
  GPTLthreadid[t] = t;

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

#ifdef ENABLE_NESTEDOMP
#ifdef INLINE_THREADING
inline
#endif
void GPTLget_nested_thread_nums (int *major, int *minor)
{
  if (omp_get_max_active_levels () > 1) { // nesting is "enabled", though not necessarily active
    volatile const int lvl = omp_get_active_level (); // lvl=2 => inside 2 #pragma omp regions
    if (lvl == 1) {
      *major = omp_get_thread_num ();
    } else if (lvl == 2) {
      *major = omp_get_ancestor_thread_num (1);
      *minor = omp_get_thread_num ();
    }
  }
}
#endif

  void GPTLprint_threadmapping (FILE *fp)
{
  int t;
  fprintf (fp, "\n");
  fprintf (fp, "Thread mapping:\n");
  for (t = 0; t < GPTLnthreads; ++t)
    fprintf (fp, "GPTLthreadid[%d] = %d\n", t, GPTLthreadid[t]);
}
