#include "private.h"

static int lock_mutex (void);      /* lock a mutex for entry into a critical region */
static int unlock_mutex (void);    /* unlock a mutex for exit from a critical region */

#if ( defined THREADED_OMP )

#include <omp.h>
static omp_lock_t lock;

int threadinit (int *nthreads, int *maxthreads)
{
  /*
  ** Must call init_lock before using the lock (get_thread_num())
  */

  omp_init_lock (&lock);

  /*
  ** In OMP case, maxthreads and nthreads are the same number
  */

  *maxthreads = omp_get_max_threads ();
  *nthreads = *maxthreads;
  if (get_thread_num (nthreads, maxthreads) > 0)
    return GPTerror ("GPTthreadinit: MUST be called only by master thread");
  return 0;
}

int get_thread_num (int *nthreads, int *maxthreads)
{
  int mythread;

  if ((mythread = omp_get_thread_num ()) >= *nthreads)
    return GPTerror ("get_thread_num: returned id %d exceed numthreads %d\n",
		     mythread, *nthreads);

  return mythread;
}

int lock_mutex (void)
{
  omp_set_lock (&lock);
  return 0;
}

static int unlock_mutex (void)
{
  omp_unset_lock (&lock);
  return 0;
}

#elif ( defined THREADED_PTHREADS )

#define MAX_THREADS 128

#include <pthread.h>

pthread_mutex_t t_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_t *threadid;

int threadinit (int *nthreads, int *maxthreads)
{
  int nbytes;

  /*
  ** Manage the threadid array which maps
  ** physical thread id's to logical id's
  */

  nbytes = MAX_THREADS * sizeof (pthread_t);
  if ( ! (threadid = (pthread_t *) malloc (nbytes)))
    return GPTerror ("threadinit: malloc failure for %d items\n", MAX_THREADS);

  /*
  ** Initialize nthreads to 1 and define the threadid array now that initialization 
  ** is done. The actual value will be determined as get_thread_num is called.
  */

  threadid[0] = pthread_self ();
  *nthreads = 1;
  *maxthreads = MAX_THREADS;

  return 0;
}

int get_thread_num (int *nthreads, int *maxthreads)
{
  int n;                 /* return value: loop index over number of threads */
  pthread_t mythreadid;  /* thread id from pthreads library */

  mythreadid = pthread_self ();

  if (lock_mutex () < 0)
    return GPTerror ("get_thread_num: mutex lock failure\n");

  /*
  ** Loop over known physical thread id's.  When my id is found, map it 
  ** to logical thread id for indexing.  If not found return a negative 
  ** number.
  ** A critical region is necessary because acess to
  ** the array threadid must be by only one thread at a time.
  */

  for (n = 0; n < *nthreads; n++)
    if (pthread_equal (mythreadid, threadid[n]))
      break;

  /*
  ** If our thread id is not in the known list, add to it after checking that
  ** we do not have too many threads.
  */

  if (n == *nthreads) {
    if (*nthreads >= MAX_THREADS)
      return GPTerror ("get_thread_num: nthreads=%d is too big Recompile "
		      "with larger value of MAX_THREADS\n", *nthreads);
    
    threadid[n] = mythreadid;
    ++*nthreads;
  }
    
  if (unlock_mutex () < 0)
    return GPTerror ("get_thread_num: mutex unlock failure\n");

  return n;
}

static int lock_mutex ()
{
  if (pthread_mutex_lock (&t_mutex) != 0)
    return GPTerror ("pthread_mutex_lock failure\n");
  return 0;
}

static int unlock_mutex ()
{
  if (pthread_mutex_unlock (&t_mutex) != 0)
    return GPTerror ("pthread_mutex_unlock failure\n");
  return 0;
}

#else

/*
** Unthreaded case
*/

int threadinit (int *nthreads, int *maxthreads)
{
  *nthreads = 1;
  *maxthreads = 1;
  return 0;
}

int get_thread_num (int *nthreads, int *maxthreads)
{
  return 0;
}
#endif
