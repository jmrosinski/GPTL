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
#include <stdio.h>

volatile int GPTLnthreads = -1;        // num threads: init to bad value
volatile int GPTLmax_threads = -1;     // max num threads
// make threadid non-static due to this file possibly being inlined
int GPTLthreadid = -1;

int GPTLthreadinit (void)
{
  static const char *thisfunc = "threadinit";

  if (GPTLnthreads != -1)
    return GPTLerror ("GPTL: Unthreaded %s: MUST only be called once", thisfunc);

  GPTLnthreads = 0;
  GPTLmax_threads = 1;
  return 0;
}

// GPTLthreadfinalize: clean up by resetting GPTLthreadid to "uninitialized" value
void GPTLthreadfinalize () {GPTLthreadid = -1;}

/*
** GPTLget_thread_num: Determine thread number of the calling thread
**                     Start PAPI counters if enabled and first call for this thread.
**
** Output results:
**   GPTLnthreads:     Number of threads (always 1)
**   GPTLthreadid:         Our thread id (always 0)
*/
#ifdef INLINE_THREADING
inline
#endif
int GPTLget_thread_num ()
{
#ifdef HAVE_PAPI
  static const char *thisfunc = "GPTLget_thread_num";
  // When HAVE_PAPI is true, if 1 or more PAPI events are enabled,
  // create and start an event set for the new thread.
  if (GPTLthreadid == -1 && GPTLget_npapievents () > 0) {
    if (GPTLcreate_and_start_events (0) < 0)
      return GPTLerror ("GPTL: Unthreaded %s: error from GPTLcreate_and_start_events for thread %0\n",
                        thisfunc);
  }
#endif

  GPTLnthreads = 1;
  GPTLthreadid = 0;
  return GPTLthreadid;
}

void GPTLprint_threadmapping (FILE *fp)
{
  fprintf (fp, "\n");
  fprintf (fp, "GPTLthreadid[0] = 0\n");
}
