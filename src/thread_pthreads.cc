/*
**
** Author: Jim Rosinski
** 
** Utility functions for pthreads-based threading
*/
#include "config.h"   // Must be first include
#include "thread.h"
#include "private.h"
#include "gptl_papi.h"
#include <stdio.h>
#include <stdlib.h>   // free
#include <pthread.h>

volatile int GPTLnthreads = -1;        // num threads: init to bad value
volatile pthread_t *GPTLthreadid = NULL;   // array of thread ids
// Set default GPTLmax_threads to a big number.
// But the user can specify GPTLmax_threads with a GPTLsetoption call.
volatile int GPTLmax_threads = 64;
#define MUTEX_API
#ifdef MUTEX_API
static volatile pthread_mutex_t t_mutex;
#else
static volatile pthread_mutex_t t_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

static int lock_mutex (void);          // lock a mutex for entry into a critical region
static int unlock_mutex (void);        // unlock a mutex for exit from a critical region

/*
** GPTLthreadinit: Allocate GPTLthreadid and initialize to -1; set max number of threads;
**             Initialize the mutex for later use; Initialize GPTLnthreads to 0
**
** Output results:
**   GPTLnthreads:   number of threads (init to zero here, increment later in GPTLget_thread_num)
**   GPTLmaxthreads: max number of threads (MAX_THREADS)
**
**   GPTLthreadid[] is allocated and initialized to -1
**   mutex is initialized for future use
**
** Return value: 0 (success) or GPTLerror (failure)
*/
int GPTLthreadinit (void)
{
  int t;
  int ret;
  static const char *thisfunc = "GPTLthreadinit";

  // The following test is not rock-solid, but it's pretty close in terms of guaranteeing that 
  // GPTLthreadinit gets called by only 1 thread. Problem is, mutex hasn't yet been initialized
  // so we can't use it.
  if (GPTLnthreads == -1)
    GPTLnthreads = 0;
  else
    return GPTLerror ("GPTL: PTHREADS %s: has already been called.\n"
                      "Maybe mistakenly called by multiple threads?\n", thisfunc);

  /*
  ** Initialize the mutex required for critical regions.
  ** Previously, t_mutex = PTHREAD_MUTEX_INITIALIZER on the static declaration line was
  ** adequate to initialize the mutex. But this failed in programs that invoked
  ** GPTLfinalize() followed by GPTLinitialize().
  ** "man pthread_mutex_init" indicates that passing NULL as the second argument to 
  ** pthread_mutex_init() should appropriately initialize the mutex, assuming it was
  ** properly destroyed by a previous call to pthread_mutex_destroy();
  */
#ifdef MUTEX_API
  if ((ret = pthread_mutex_init ((pthread_mutex_t *) &t_mutex, NULL)) != 0)
    return GPTLerror ("GPTL: PTHREADS %s: mutex init failure: ret=%d\n", thisfunc, ret);
#endif
  
  // GPTLmax_threads is either its default initialization value, or set by a user
  // call to GPTLsetoption().
  // Allocate the GPTLthreadid array which maps physical thread IDs to logical IDs
  if (GPTLthreadid) 
    return GPTLerror ("GPTL: PTHREADS %s: GPTLthreadid not null\n", thisfunc);
  else if ( ! (GPTLthreadid = (pthread_t *) GPTLallocate (GPTLmax_threads * sizeof (pthread_t), thisfunc)))
    return GPTLerror ("GPTL: PTHREADS %s: malloc failure for %d elements of GPTLthreadid\n", 
                      thisfunc, GPTLmax_threads);

  // Initialize GPTLthreadid array to flag values for use by GPTLget_thread_num().
  // GPTLget_thread_num() will fill in the values on first use.
  for (t = 0; t < GPTLmax_threads; ++t)
    GPTLthreadid[t] = (pthread_t) -1;
#ifdef VERBOSE
  printf ("GPTL: PTHREADS %s: Set GPTLmax_threads=%d GPTLnthreads=%d\n",
	  thisfunc, GPTLmax_threads, GPTLnthreads);
#endif
  return 0;
}

/*
** threadfinalize: Clean up
**
** Output results:
**   GPTLthreadid array is freed and array pointer nullified
**   mutex is destroyed
*/
void GPTLthreadfinalize ()
{
  int ret;

#ifdef MUTEX_API
  if ((ret = pthread_mutex_destroy ((pthread_mutex_t *) &t_mutex)) != 0)
    printf ("GPTL: threadfinalize: failed attempt to destroy t_mutex: ret=%d\n", ret);
#endif
  free ((void *) GPTLthreadid);
  GPTLthreadid = 0;
}

/*
** GPTLget_thread_num: Determine zero-based thread number of the calling thread.
**                 Update GPTLnthreads and GPTLmax_threads if necessary.
**                 Start PAPI counters if enabled and first call for this thread.
**
** Output results:
**   GPTLnthreads: Updated number of threads
**   GPTLthreadid: Our thread id added to list on 1st call 
**
** Return value: thread number (success) or GPTLerror (failure)
*/
#ifdef INLINE_THREADING
inline
#endif
int GPTLget_thread_num (void)
{
  int t;                   // logical thread number, defined by array index of found GPTLthreadid
  pthread_t mythreadid;    // thread id from pthreads library
  int retval = -1;         // value to return to caller: init to bad value to please compiler
  bool foundit = false;    // thread id found in list
  static const char *thisfunc = "GPTLget_thread_num";

  mythreadid = pthread_self ();

  // If our thread number has already been set in the list, we are done
  // VECTOR code should run a bit faster on vector machines.
#define VECTOR
#ifdef VECTOR
  for (t = 0; t < GPTLnthreads; ++t)
    if (pthread_equal (mythreadid, GPTLthreadid[t])) {
      foundit = true;
      retval = t;
    }

  if (foundit)
    return retval;
#else
  for (t = 0; t < GPTLnthreads; ++t)
    if (pthread_equal (mythreadid, GPTLthreadid[t]))
      return t;
#endif

  // Thread id not found. Define a critical region, then start PAPI counters if
  // necessary and modify GPTLthreadid[] with our id.
  if (lock_mutex () < 0)
    return GPTLerror ("GPTL: PTHREADS %s: mutex lock failure\n", thisfunc);

  // If our thread id is not in the known list, add to it after checking that
  // we do not have too many threads.
  if (GPTLnthreads >= GPTLmax_threads) {
    if (unlock_mutex () < 0)
      fprintf (stderr, "GPTL: UNDERLYING_PTHREADS %s: mutex unlock failure\n", thisfunc);

    return GPTLerror ("GPTL: UNDERLYING_PTHREADS %s: thread index=%d is too big. Need to invoke \n"
		      "GPTLsetoption(GPTLmax_threads,value) or recompile GPTL with a\n"
		      "larger value of MAX_THREADS\n", thisfunc, GPTLnthreads);
  }

  GPTLthreadid[GPTLnthreads] = mythreadid;

#ifdef VERBOSE
  printf ("GPTL: PTHREADS %s: 1st call GPTLthreadid=%lu maps to location %d\n", 
          thisfunc, (unsigned long) mythreadid, GPTLnthreads);
#endif

#ifdef HAVE_PAPI

  // When HAVE_PAPI is true, if 1 or more PAPI events are enabled,
  // create and start an event set for the new thread.
  if (GPTLget_npapievents () > 0) {
#ifdef VERBOSE
    printf ("GPTL: PTHREADS %s: Starting EventSet GPTLthreadid=%lu location=%d\n", 
            thisfunc, (unsigned long) mythreadid, GPTLnthreads);
#endif
    if (GPTLcreate_and_start_events (GPTLnthreads) < 0) {
      if (unlock_mutex () < 0)
        fprintf (stderr, "GPTL: PTHREADS %s: mutex unlock failure\n", thisfunc);
      
      return GPTLerror ("GPTL: PTHREADS %s: error from GPTLcreate_and_start_events thread=%d\n", 
                        thisfunc, GPTLnthreads);
    }
  }
#endif

  /*
  ** IMPORTANT to set return value before unlocking the mutex!!!!
  ** "return GPTLnthreads-1" fails occasionally when another thread modifies
  ** GPTLnthreads after it gets the mutex!
  */
  retval = GPTLnthreads++;

#ifdef VERBOSE
  printf ("GPTL: PTHREADS %s: GPTLnthreads bumped to %d\n", thisfunc, GPTLnthreads);
#endif

  if (unlock_mutex () < 0)
    return GPTLerror ("GPTL: PTHREADS %s: mutex unlock failure\n", thisfunc);

  return retval;
}

// lock_mutex: lock a mutex for private access
static int lock_mutex ()
{
  static const char *thisfunc = "lock_mutex";

  if (pthread_mutex_lock ((pthread_mutex_t *) &t_mutex) != 0)
    return GPTLerror ("GPTL: %s: failure from pthread_mutex_lock\n", thisfunc);

  return 0;
}

// unlock_mutex: unlock a mutex from private access
static int unlock_mutex ()
{
  static const char *thisfunc = "unlock_mutex";

  if (pthread_mutex_unlock ((pthread_mutex_t *) &t_mutex) != 0)
    return GPTLerror ("GPTL: %s: failure from pthread_mutex_unlock\n", thisfunc);
  return 0;
}

void GPTLprint_threadmapping (FILE *fp)
{
  int t;
  fprintf (fp, "\n");
  fprintf (fp, "Thread mapping:\n");
  for (t = 0; t < GPTLnthreads; ++t)
    fprintf (fp, "GPTLthreadid[%d] = %ld\n", t, GPTLthreadid[t]);
}
