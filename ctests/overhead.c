#include <sys/time.h>  /* gettimeofday */
#include <sys/times.h> /* times */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>    /* gettimeofday */
#include <assert.h>

#ifdef THREADED_OMP
#include <omp.h>
#endif

#ifdef HAVE_PAPI
#include <papi.h>
#endif

#include "../gptl.h"

#define tablesiz 128*31

typedef struct TIMER {
  char name[32];
} Timer;

typedef struct {
  unsigned int nument;
  Timer **entries;
} Hashentry;

static Hashentry hashtable[tablesiz];    /* table of entries hashed by sum of chars */

Timer longnames_samehash[] = {
  "9123456789012345678901234567890",
  "9234567890123456789012345678901",
  "9345678901234567890123456789012",
  "9456789012345678901234567890123",
  "9567890123456789012345678901234"
};

Timer longnames_diffhash[] = {
  "0123456789012345678901234567890",
  "1234567890123456789012345678901",
  "2345678901234567890123456789012",
  "3456789012345678901234567890123",
  "4567890123456789012345678901234"
};

Timer shortnames_samehash[] = {
  "mn",
  "lo",
  "kp",
  "jq",
  "ir"
};

Timer shortnames_diffhash[] = {
  "ab",
  "cd",
  "ef",
  "gh",
  "ij"
};

static int numstr = sizeof (shortnames_diffhash) / sizeof (Timer);

inline static void *getentry (const Hashentry *, const char *, int *);
static void overhead (int, int);
int gethashvalue (char *);

int main (int argc, char **argv)
{
  int nompiter;
  int ompiter;
  int ninvoke;
  int papiopt;
  int gptlopt;
  int val;

#ifdef NUMERIC_TIMERS
  printf ("%s not enabled for NUMERIC_TIMERS\n", argv[0]);
  exit (-1);
#else

#ifdef HAVE_PAPI
  GPTLPAPIprinttable ();
  while (1) {
    printf ("Enter PAPI option to enable, non-negative number when done\n");
    scanf ("%d", &papiopt);
    if (papiopt >= 0)
      break;
    if (GPTLsetoption (papiopt, 1) < 0)
      printf ("gptlsetoption failure\n");
  }
#endif

  printf ("GPTLwall           = 1\n");
  printf ("GPTLcpu            = 2\n");
  printf ("GPTLabort_on_error = 3\n");
  printf ("GPTLoverhead       = 4\n");
  while (1) {
    printf ("Enter GPTL option and enable flag, negative numbers when done\n");
    scanf ("%d %d", &gptlopt, &val);
    if (gptlopt <= 0)
      break;
    if (GPTLsetoption (gptlopt, val) < 0)
      printf ("gptlsetoption failure\n");
  }

  printf ("Enter number of iterations for threaded loop:\n");
  scanf ("%d", &nompiter);
  printf ("Enter number of invocations of overhead portion:\n");
  scanf ("%d", &ninvoke);
  printf ("nompiter=%d ninvoke=%d\n", nompiter, ninvoke);

  GPTLinitialize ();
  GPTLstart ("total");

#ifdef THREADED_OMP
#pragma omp parallel for private (ompiter)
#endif

  for (ompiter = 0; ompiter < nompiter; ++ompiter) {
    overhead (ompiter, ninvoke);
  }
  GPTLstop ("total");
  GPTLpr (0);
  GPTLfinalize ();
#endif
}

#ifndef NUMERIC_TIMERS
static void overhead (int iter, int ninvoke)
{
  int i;
#ifdef THREADED_OMP
  int tnum;
#endif
  int indx;
  char **strings1;
  char **strings2;
  struct timeval tp;
  struct tms buf;
  void *nothing;

#ifdef THREADED_OMP
  GPTLstart ("get_thread_num");
  for (i = 0; i < ninvoke; ++i) {
    tnum = omp_get_thread_num ();
  }
  GPTLstop ("get_thread_num");
#endif

  GPTLstart ("gettimeofday");
  for (i = 0; i < ninvoke; ++i) {
    gettimeofday (&tp, 0);
  }
  GPTLstop ("gettimeofday");

  GPTLstart ("times");
  for (i = 0; i < ninvoke; ++i) {
    (void) times (&buf);
  }
  GPTLstop ("times");
  
  struct timeval tp1, tp2;      /* argument returned from gettimeofday */
  float overhead;

  gettimeofday (&tp1, 0);
  for (i = 0; i < ninvoke/10; ++i) {
    GPTLstart ("overhead_est.");
    GPTLstop ("overhead_est.");
    GPTLstart ("overhead_est.");
    GPTLstop ("overhead_est.");
    GPTLstart ("overhead_est.");
    GPTLstop ("overhead_est.");
    GPTLstart ("overhead_est.");
    GPTLstop ("overhead_est.");
    GPTLstart ("overhead_est.");
    GPTLstop ("overhead_est.");
    GPTLstart ("overhead_est.");
    GPTLstop ("overhead_est.");
    GPTLstart ("overhead_est.");
    GPTLstop ("overhead_est.");
    GPTLstart ("overhead_est.");
    GPTLstop ("overhead_est.");
    GPTLstart ("overhead_est.");
    GPTLstop ("overhead_est.");
    GPTLstart ("overhead_est.");
    GPTLstop ("overhead_est.");
  }
  gettimeofday (&tp2, 0);
  overhead = (tp2.tv_sec  - tp1.tv_sec) + 
       1.e-6*(tp2.tv_usec - tp1.tv_usec);
  printf ("overhead = %f sec.  Compare to 'overhead_est' in timing output\n", overhead);

  Timer **eptr;
  int nument;

  for (i = 0; i < numstr; i++) {
    indx = gethashvalue (shortnames_diffhash[i].name);
    assert (indx < tablesiz);
    ++hashtable[indx].nument;
    nument = hashtable[indx].nument;
    eptr = (Timer **) realloc (hashtable[indx].entries, nument * sizeof (Timer *));
    hashtable[indx].entries           = eptr;
    hashtable[indx].entries[nument-1] = &shortnames_diffhash[i];

    indx = gethashvalue (shortnames_samehash[i].name);
    assert (indx < tablesiz);
    ++hashtable[indx].nument;
    nument = hashtable[indx].nument;
    eptr = (Timer **) realloc (hashtable[indx].entries, nument * sizeof (Timer *));
    hashtable[indx].entries           = eptr;
    hashtable[indx].entries[nument-1] = &shortnames_samehash[i];

    indx = gethashvalue (longnames_diffhash[i].name);
    assert (indx < tablesiz);
    ++hashtable[indx].nument;
    nument = hashtable[indx].nument;
    eptr = (Timer **) realloc (hashtable[indx].entries, nument * sizeof (Timer *));
    hashtable[indx].entries           = eptr;
    hashtable[indx].entries[nument-1] = &longnames_diffhash[i];

    indx = gethashvalue (longnames_samehash[i].name);
    assert (indx < tablesiz);
    ++hashtable[indx].nument;
    nument = hashtable[indx].nument;
    eptr = (Timer **) realloc (hashtable[indx].entries, nument * sizeof (Timer *));
    hashtable[indx].entries           = eptr;
    hashtable[indx].entries[nument-1] = &longnames_samehash[i];
  }
    
  GPTLstart ("getentry_shortnames_diffhash");
  for (i = 0; i < ninvoke; ++i) {
    nothing = getentry (hashtable, "mn", &indx);
    assert (indx < tablesiz);
    if ( ! nothing) 
      printf ("error: index not found");
  }
  GPTLstop ("getentry_shortnames_diffhash");

  GPTLstart ("getentry_shortnames_samehash");
  for (i = 0; i < ninvoke; ++i) {
    nothing = getentry (hashtable, "mn", &indx);
    assert (indx < tablesiz);
    if ( ! nothing) 
      printf ("error: index not found");
  }
  GPTLstop ("getentry_shortnames_samehash");

  GPTLstart ("getentry_longnames_diffhash");
  for (i = 0; i < ninvoke; ++i) {
    nothing = getentry (hashtable, "4567890123456789012345678901234", &indx);
    assert (indx < tablesiz);
    if ( ! nothing) 
      printf ("error: index not found");
  }
  GPTLstop ("getentry_longnames_diffhash");

  GPTLstart ("getentry_longnames_samehash");
  for (i = 0; i < ninvoke; ++i) {
    nothing = getentry (hashtable, "9567890123456789012345678901234", &indx);
    assert (indx < tablesiz);
    if ( ! nothing) 
      printf ("error: index not found");
  }
  GPTLstop ("getentry_longnames_samehash");
}

static inline void *getentry (const Hashentry *hashtable,
			      const char *name, 
			      int *indx)
{
  int i;

  const char *c = name;

  for (*indx = 0; *c; c++)
    *indx += *c;

  for (i = 0; i < hashtable[*indx].nument; i++)
    if (strcmp (name, hashtable[*indx].entries[i]->name) == 0)
      return hashtable[*indx].entries[i];

  return 0;
}

int gethashvalue (char *name)
{
  int i;
  int indx;

  const char *c = name;

  for (indx = 0; *c; c++)
    indx += *c;

  return indx;
}
#endif
