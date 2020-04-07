#include <pthread.h>

#define MUTEX_API
#ifdef MUTEX_API
static volatile pthread_mutex_t t_mutex;
#else
static volatile pthread_mutex_t t_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif
volatile pthread_t *GPTLthreadid = 0;  /* array of thread ids */
static int lock_mutex (void);          /* lock a mutex for entry into a critical region */
static int unlock_mutex (void);        /* unlock a mutex for exit from a critical region */

// TODO: make maxthreads variable
static volatile int maxthreads = 64;

/*
** threadinit: Allocate GPTLthreadid and initialize to -1; set max number of threads;
**             Initialize the mutex for later use; Initialize nthreads to 0
**
** Output results:
**   nthreads:   number of threads (init to zero here, increment later in get_thread_num)
**   maxthreads: max number of threads (MAX_THREADS)
**
**   GPTLthreadid[] is allocated and initialized to -1
**   mutex is initialized for future use
**
** Return value: 0 (success) or GPTLerror (failure)
*/
static int threadinit (void)
{
  int t;        /* thread number */
  int ret;      /* return code */
  static const char *thisfunc = "threadinit";

  /*
  ** The following test is not rock-solid, but it's pretty close in terms of guaranteeing that 
  ** threadinit gets called by only 1 thread. Problem is, mutex hasn't yet been initialized
  ** so we can't use it.
  */
  if (nthreads == -1)
    nthreads = 0;
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
  
  /* maxthreads is either its default initialization value, or set by a user
  ** call to GPTLsetoption().
  ** Allocate the threadid array which maps physical thread IDs to logical IDs
  */
  if (GPTLthreadid) 
    return GPTLerror ("GPTL: PTHREADS %s: GPTLthreadid not null\n", thisfunc);
  else if ( ! (GPTLthreadid = (pthread_t *) GPTLallocate (maxthreads * sizeof (pthread_t), thisfunc)))
    return GPTLerror ("GPTL: PTHREADS %s: malloc failure for %d elements of GPTLthreadid\n", 
                      thisfunc, maxthreads);

  /*
  ** Initialize GPTLthreadid array to flag values for use by get_thread_num().
  ** get_thread_num() will fill in the values on first use.
  */
  for (t = 0; t < maxthreads; ++t)
    GPTLthreadid[t] = (pthread_t) -1;

#ifdef VERBOSE
  printf ("GPTL: PTHREADS %s: Set maxthreads=%d nthreads=%d\n", thisfunc, maxthreads, nthreads);
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
static void threadfinalize ()
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
** get_thread_num: Determine zero-based thread number of the calling thread.
**                 Update nthreads and maxthreads if necessary.
**                 Start PAPI counters if enabled and first call for this thread.
**
** Output results:
**   nthreads: Updated number of threads
**   GPTLthreadid: Our thread id added to list on 1st call 
**
** Return value: thread number (success) or GPTLerror (failure)
*/
static inline int get_thread_num (void)
{
  int t;                   /* logical thread number, defined by array index of found GPTLthreadid */
  pthread_t mythreadid;    /* thread id from pthreads library */
  int retval = -1;         /* value to return to caller: init to bad value to please compiler */
  bool foundit = false;    /* thread id found in list */
  static const char *thisfunc = "get_thread_num";

  mythreadid = pthread_self ();

  /*
  ** If our thread number has already been set in the list, we are done
  ** VECTOR code should run a bit faster on vector machines.
  */
#define VECTOR
#ifdef VECTOR
  for (t = 0; t < nthreads; ++t)
    if (pthread_equal (mythreadid, GPTLthreadid[t])) {
      foundit = true;
      retval = t;
    }

  if (foundit)
    return retval;
#else
  for (t = 0; t < nthreads; ++t)
    if (pthread_equal (mythreadid, GPTLthreadid[t]))
      return t;
#endif

  /* 
  ** Thread id not found. Define a critical region, then start PAPI counters if
  ** necessary and modify GPTLthreadid[] with our id.
  */
  if (lock_mutex () < 0)
    return GPTLerror ("GPTL: PTHREADS %s: mutex lock failure\n", thisfunc);

  /*
  ** If our thread id is not in the known list, add to it after checking that
  ** we do not have too many threads.
  */
  if (nthreads >= maxthreads) {
    if (unlock_mutex () < 0)
      fprintf (stderr, "GPTL: PTHREADS %s: mutex unlock failure\n", thisfunc);

    return GPTLerror ("GPTL: THREADED_PTHREADS %s: thread index=%d is too big. Need to invoke \n"
		      "GPTLsetoption(GPTLmaxthreads,value) or recompile GPTL with a\n"
		      "larger value of MAX_THREADS\n", thisfunc, nthreads);
  }

  GPTLthreadid[nthreads] = mythreadid;

#ifdef VERBOSE
  printf ("GPTL: PTHREADS %s: 1st call GPTLthreadid=%lu maps to location %d\n", 
          thisfunc, (unsigned long) mythreadid, nthreads);
#endif

#ifdef HAVE_PAPI

  /*
  ** When HAVE_PAPI is true, if 1 or more PAPI events are enabled,
  ** create and start an event set for the new thread.
  */
  if (gptl_papi::npapievents () > 0) {
#ifdef VERBOSE
    printf ("GPTL: PTHREADS %s: Starting EventSet GPTLthreadid=%lu location=%d\n", 
            thisfunc, (unsigned long) mythreadid, nthreads);
#endif
    if (GPTLcreate_and_start_events (nthreads) < 0) {
      if (unlock_mutex () < 0)
        fprintf (stderr, "GPTL: PTHREADS %s: mutex unlock failure\n", thisfunc);

      return GPTLerror ("GPTL: PTHREADS %s: error from GPTLcreate_and_start_events for thread %d\n", 
                        thisfunc, nthreads);
    }
  }
#endif

  /*
  ** IMPORTANT to set return value before unlocking the mutex!!!!
  ** "return nthreads-1" fails occasionally when another thread modifies
  ** nthreads after it gets the mutex!
  */
  retval = nthreads++;

#ifdef VERBOSE
  printf ("GPTL: PTHREADS %s: nthreads bumped to %d\n", thisfunc, nthreads);
#endif

  if (unlock_mutex () < 0)
    return GPTLerror ("GPTL: PTHREADS %s: mutex unlock failure\n", thisfunc);

  return retval;
}

/*
** lock_mutex: lock a mutex for private access
*/
static int lock_mutex ()
{
  static const char *thisfunc = "lock_mutex";

  if (pthread_mutex_lock ((pthread_mutex_t *) &t_mutex) != 0)
    return GPTLerror ("GPTL: %s: failure from pthread_lock_mutex\n", thisfunc);

  return 0;
}

/*
** unlock_mutex: unlock a mutex from private access
*/
static int unlock_mutex ()
{
  static const char *thisfunc = "unlock_mutex";

  if (pthread_mutex_unlock ((pthread_mutex_t *) &t_mutex) != 0)
    return GPTLerror ("GPTL: %s: failure from pthread_unlock_mutex\n", thisfunc);
  return 0;
}

static void print_threadmapping (FILE *fp)
{
  int t;

  fprintf (fp, "\n");
  fprintf (fp, "Thread mapping:\n");
  for (t = 0; t < nthreads; ++t)
    fprintf (fp, "GPTLthreadid[%d] = %lu\n", t, (unsigned long) GPTLthreadid[t]);
}

