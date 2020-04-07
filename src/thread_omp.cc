#include "config.h" // Must be first include
#include "thread.h"
#include "util.h"
#include "gptl_papi.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static volatile int *threadid = NULL; // array of thread ids

namespace gptl_thread {
  volatile int maxthreads = -1;       // max threads
  volatile int nthreads = -1;         // num threads. Init to bad value

  extern "C" {
    /*
    ** threadinit: Allocate and initialize threadid; set max number of threads
    **
    ** Output results:
    **   maxthreads: max number of threads
    **
    **   threadid[] is allocated and initialized to -1
    **
    **
    ** Return value: 0 (success) or GPTLerror (failure)
    */
    int threadinit ()
    {
      using namespace gptl_util;
      int t;  // loop index
      static const char *thisfunc = "threadinit";
      
      if (omp_get_thread_num () != 0)
	return error ("OMP %s: MUST only be called by the master thread\n", thisfunc);

      // Allocate the threadid array which maps physical thread IDs to logical IDs 
      // For OpenMP this will be just threadid[iam] = iam;
      if (threadid) 
	return error ("OMP %s: has already been called.\nMaybe mistakenly called by multiple threads?",
		      thisfunc);

      // maxthreads may have been set by the user, in which case use that. But if as 
      // yet uninitialized, set to the current value of OMP_NUM_THREADS. 
      if (maxthreads == -1)
	maxthreads = MAX ((1), (omp_get_max_threads ()));

      if ( ! (threadid = (int *) gptl_util::allocate (maxthreads * sizeof (int), thisfunc)))
	return gptl_util::error ("OMP %s: malloc failure for %d elements of threadid\n",
				 thisfunc, maxthreads);

      // Initialize threadid array to flag values for use by get_thread_num().
      // get_thread_num() will fill in the values on first use.
      for (t = 0; t < maxthreads; ++t)
	threadid[t] = -1;
#ifdef VERBOSE
      printf ("GPTL: OMP %s: Set maxthreads=%d\n", thisfunc, maxthreads);
#endif
      return 0;
    }

    void threadfinalize ()
    {
      free ((void *) threadid);
      threadid = 0;
    }

    /*
    ** get_thread_num: Determine thread number of the calling thread
    **                 Start PAPI counters if enabled and first call for this thread.
    **
    ** Output results:
    **   nthreads:     Number of threads
    **   threadid: Our thread id added to list on 1st call
    **
    ** Return value: thread number (success) or gptl_util::error (failure)
    **   5/8/16: Modified to enable 2-level OMP nesting: Fold combination of current and parent
    **   thread info into a single index
    */
    inline int get_thread_num ()
    {
      using namespace gptl_util;
      int t;        // thread number
      static const char *thisfunc = "get_thread_num";

#ifdef ENABLE_NESTEDOMP
      int myid;            // my thread id
      int lvl;             // nest level: Currently only 2 nesting levels supported
      int parentid;        // thread number of parent team
      int my_nthreads;     // number of threads in the parent team

      myid = omp_get_thread_num ();
      if (omp_get_nested ()) {         // nesting is "enabled", though not necessarily active
	lvl = omp_get_active_level (); // lvl=2 => inside 2 #pragma omp region
	if (lvl < 2) {
	  // 0 or 1-level deep: simply use thread id as index
	  t = myid;
	} else if (lvl == 2) {
	  // Create a unique id "t" for indexing into singly-dimensioned thread array
	  parentid    = omp_get_ancestor_thread_num (lvl-1);
	  my_nthreads = omp_get_team_size (lvl);
	  t           = parentid*my_nthreads + myid;
	} else {
	  return error ("OMP %s: GPTL supports only 2 nested OMP levels got %d\n", thisfunc, lvl);
	}
      } else {
	// un-nested case: thread id is index
	t = myid;
      }
#else
      t = omp_get_thread_num ();
#endif
      if (t >= maxthreads)
	return error ("OMP %s: returned id=%d exceeds maxthreads=%d\n", thisfunc, t, maxthreads);

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
      if (gptl_papi::npapievents () > 0) {
#ifdef VERBOSE
	printf ("GPTL: OMP %s: Starting EventSet t=%d\n", thisfunc, t);
#endif

	if (create_and_start_events (t) < 0)
	  return error ("GPTL: OMP %s: error from GPTLcreate_and_start_events for thread %d\n", 
			thisfunc, t);
      }
#endif

      // nthreads = maxthreads based on setting in threadinit or user call to GPTLsetoption()
      nthreads = maxthreads;
#ifdef VERBOSE
      printf ("GPTL: OMP %s: nthreads=%d\n", thisfunc, nthreads);
#endif

      return t;
    }

    void print_threadmapping (FILE *fp)
    {
      fprintf (fp, "\n");
      fprintf (fp, "Thread mapping:\n");
      for (int t = 0; t < nthreads; ++t)
	fprintf (fp, "threadid[%d] = %d\n", t, threadid[t]);
    }
  }
}
