/*
**
** Author: Jim Rosinski
** 
** Utility functions for placeholder threading, plus start PAPI events if enabled
*/

#include "config.h"   // Must be first include
#include "thread.h"
#include "private.h"
#include "gptl_papi.h"
#include "util.h"

#include <stdio.h>

namespace thread {
  volatile int max_threads = -1;     // max num threads
  volatile int nthreads = -1;        // num threads: init to bad value
  volatile int *threadid = NULL;

  extern "C" int threadinit (void)
  {
    static const char *thisfunc = "threadinit";

    if (threadid) 
      return util::error ("unthreaded %s: has already been called.\n", thisfunc);

    max_threads = 1;
    
    if (nthreads != -1)
      return util::error ("GPTL: Unthreaded %s: MUST only be called once", thisfunc);

    if ( ! (threadid = (int *) util::allocate (max_threads * sizeof (int), thisfunc)))
      return util::error ("Unthreaded %s: malloc failure for %d elements of threadid\n",
			  thisfunc, max_threads);

    nthreads = 1;
    return 0;
  }

  // GPTLthreadfinalize: clean up by resetting GPTLthreadid to "uninitialized" value
  extern "C" void threadfinalize () {*threadid = -1;}

  /*
  ** get_thread_num: Determine thread number of the calling thread
  **                 Start PAPI counters if enabled and first call for this thread.
  **
  ** Output results:
  **   nthreads:     Number of threads (always 1)
  **   threadid:         Our thread id (always 0)
  */
  extern "C" int get_thread_num ()
  {
#ifdef HAVE_PAPI
    static const char *thisfunc = "GPTLget_thread_num";
    // When HAVE_PAPI is true, if 1 or more PAPI events are enabled,
    // create and start an event set for the new thread.
    if (*threadid == -1 && GPTLget_npapievents () > 0) {
      if (create_and_start_events (0) < 0)
	return util::error ("GPTL: Unthreaded %s: error from create_and_start_events for thread %0\n",
			    thisfunc);
  }
#endif

    nthreads = 1;
    *threadid = 0;
    return *threadid;
  }

  extern "C" void print_threadmapping (FILE *fp)
  {
    fprintf (fp, "\n");
    fprintf (fp, "threadid[0] = 0\n");
  }
}
