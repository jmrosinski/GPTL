#include <stdio.h>
#include <string.h>

#ifdef HAVE_PAPI

#include <stdlib.h>
#include "../private.h"
#include <papi.h>

#if ( defined THREADED_OMP )
#include <omp.h>
#elif ( defined THREADED_PTHREADS )
#include <pthread.h>
#endif

#define MAX_AUX 16
static Entry eventlist[MAX_AUX];
static int nevents = 0;                 /* number of events: initialize to 0 */ 
static int *EventSet;
static long long **papicounters;
static long long **prvcounters;
static bool *started;
int nthreads;
int maxthreads;

void parsub (int);

int main ()
{
  int ret;
  int counter;
  int n;
  int i;
  int ompiter, nompiter;
  char papiname[PAPI_MAX_STR_LEN];

  if ((ret = PAPI_library_init (PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) {
    printf (PAPI_strerror (ret));
    return -1;
  }

#if ( defined THREADED_OMP )
  if (PAPI_thread_init ((unsigned long (*)(void)) (omp_get_thread_num)) != PAPI_OK)
    return -2;
#elif ( defined THREADED_PTHREADS )
  if (PAPI_thread_init ((unsigned long (*)(void)) (pthread_self)) != PAPI_OK)
    return -3;
#endif

  threadinit (&nthreads, &maxthreads);

  started      = (bool *)       malloc (maxthreads * sizeof (bool));
  EventSet     = (int *)        malloc (maxthreads * sizeof (int));
  papicounters = (long_long **) malloc (maxthreads * sizeof (long_long *));
  prvcounters  = (long_long **) malloc (maxthreads * sizeof (long_long *));

  for (n = 0; n < maxthreads; n++) {
    started[n] = false;
    EventSet[n] = PAPI_NULL;
    papicounters[n] = (long_long *) malloc (MAX_AUX * sizeof (long_long));
    prvcounters[n] = (long_long *) malloc (MAX_AUX * sizeof (long_long));
    for (i = 0; i < MAX_AUX; i++)
      prvcounters[n][i] = 0;
  }

  while (1) {
    printf ("Enter option to be enabled, or 'done' when done:\n");
    scanf ("%s", &papiname);

    if (strncmp (papiname, "done", 4) == 0)
      break;

    (void) PAPI_event_name_to_code (papiname, &counter);
    if ((ret = PAPI_query_event (counter)) != PAPI_OK) {
      printf ("Event %s not available on this arch\n", papiname);
    } else {
      if (nevents+1 > MAX_AUX) {
	printf ("Event %s is too many\n", papiname);
      } else {
	eventlist[nevents].counter = counter;
	eventlist[nevents].str     = papiname;
	printf ("Event %s enabled\n", eventlist[nevents].str);
	++nevents;
      }
    }
  }

  printf ("Enter number of parallel iterations\n");
  scanf ("%d", &nompiter);

#pragma omp parallel for private (ompiter)

  for (ompiter = 0; ompiter < nompiter; ++ompiter) {
    parsub (ompiter);
  }
  return 0;
}

void parsub (int iter)
{
  int mythread;
  int ret;
  int i, n;
  float sum;

  /*
  ** get_thread_num was inlined if OMP or no threading, so this code can't just call it
  */

#if ( defined THREADED_OMP )
  mythread = omp_get_thread_num ();
#elif ( defined THREADED_PTHREADS )
  mythread = get_thread_num (&nthreads, &maxthreads);
#else
  mythread = 0;
#endif

  if ( ! started[mythread]) {
    if ((ret = PAPI_create_eventset (&EventSet[mythread])) != PAPI_OK) {
      printf ("GPTL_PAPIstart: failure creating eventset: %s\n", 
	      PAPI_strerror (ret));
      exit (1);
    }

    for (n = 0; n < nevents; n++) {
      if ((ret = PAPI_add_event (EventSet[mythread], eventlist[n].counter)) != PAPI_OK) {
	printf ("%s\n", PAPI_strerror (ret));
	printf ("Failure attempting to add event: %s\n", eventlist[n].str);
      }
    }
    if ((ret = PAPI_start (EventSet[mythread])) != PAPI_OK)
      printf ("%s\n", PAPI_strerror (ret));

    started[mythread] = true;
  }

  while (1) {
    if ((ret = PAPI_read (EventSet[mythread], papicounters[mythread])) != PAPI_OK) {
      printf ("PAPI_read error\n");
      exit (1);
    }

    for (n = 0; n < nevents; n++) {
      printf ("papicounters[%d][%d]=%ld\n", mythread, n, (long) papicounters[mythread][n]);
      if (papicounters[mythread][n] < prvcounters[mythread][n]) {
	printf ("papicounters[%d][%d]=%ld %ld\n", 
		mythread, n, (long) papicounters[mythread][n], (long) prvcounters[mythread][n]);
	exit (1);
      }
      prvcounters[mythread][n] = papicounters[mythread][n];
    }

    sum = 0;
    for (i = 0; i <= iter; i++) {
      sum += i*iter;
    }
  }
}

#else

int main ()
{
  printf ("PAPI not enabled so this code does nothing\n");
  return 0;
}

#endif
