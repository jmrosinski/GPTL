#include <stdlib.h>        /* malloc */
#include <sys/time.h>      /* gettimeofday */
#include <sys/times.h>     /* times */
#include <unistd.h>        /* gettimeofday, syscall */
#include <stdio.h>
#include <string.h>        /* memset, strcmp (via STRMATCH) */
#include <ctype.h>         /* isdigit */

#ifdef HAVE_PAPI
#include <papi.h>          /* PAPI_get_real_usec */
#endif

#ifdef UNICOSMP
#include <intrinsics.h>    /* rtc */
#endif

#if ( defined HAVE_LIBMPI ) || ( defined HAVE_LIBMPICH )
#include <mpi.h>
#endif

#ifdef HAVE_LIBRT
#include <time.h>
#endif

#include "private.h"

static Timer **timers = 0;       /* linked list of timers */
static Timer **last = 0;         /* last element in list */
static int *max_depth;           /* maximum indentation level encountered */
static int *max_name_len;        /* max length of timer name */

typedef struct {
  int depth;                     /* depth in calling tree */
  int padding[31];               /* padding is to mitigate false cache sharing */
} Nofalse; 
static Nofalse *current_depth;

static int nthreads    = -1;     /* num threads. Init to bad value */
static int maxthreads  = -1;     /* max threads (=nthreads for OMP). Init to bad value */
static int depthlimit  = 99999;  /* max depth for timers (99999 is effectively infinite) */
static bool disabled = false;    /* Timers disabled? */
static bool initialized = false; /* GPTLinitialize has been called */
static bool dousepapi = false;   /* saves a function call if stays false */
static bool verbose = true;      /* output verbosity */

static time_t ref_gettimeofday = -1; /* ref start point for gettimeofday */
static time_t ref_clock_gettime = -1;/* ref start point for clock_gettime */
static long long ref_papitime = -1;  /* ref start point for PAPI_get_real_usec */

typedef struct {
  const Option option;           /* wall, cpu, etc. */
  const char *str;               /* descriptive string for printing */
  bool enabled;                  /* flag */
} Settings;

/* Options, print strings, and default enable flags */

static Settings cpustats =      {GPTLcpu,      "Usr       sys       usr+sys   ", false};
static Settings wallstats =     {GPTLwall,     "Wallclock max       min       ", true };
static Settings overheadstats = {GPTLoverhead, "UTR Overhead  "                , true };

static Hashentry **hashtable;    /* table of entries */
static long ticks_per_sec;       /* clock ticks per second */

/* Local function prototypes */

static void printstats (const Timer *, FILE *, const int, const bool, double);
static void add (Timer *, const Timer *);
static inline int get_cpustamp (long *, long *);

#if ( ! defined THREADED_PTHREADS )
static inline int get_thread_num (int *, int *);      /* determine thread number */
#endif

/* These are the (possibly) supported underlying wallclock timers */

static inline double utr_nanotime (void);
static inline double utr_rtc (void);
static inline double utr_mpiwtime (void);
static inline double utr_clock_gettime (void);
static inline double utr_papitime (void);
static inline double utr_gettimeofday (void);

static int init_nanotime (void);
static int init_rtc (void);
static int init_mpiwtime (void);
static int init_clock_gettime (void);
static int init_papitime (void);
static int init_gettimeofday (void);

static double utr_getoverhead (void);
static inline Timer *getentry (const Hashentry *, const char *, int *);

typedef struct {
  const Funcoption option;
  double (*func)(void);
  int (*funcinit)(void);
  const char *name;
} Funcentry;

static Funcentry funclist[] = {
  {GPTLgettimeofday, utr_gettimeofday,  init_gettimeofday,  "gettimeofday"},
  {GPTLnanotime,     utr_nanotime,      init_nanotime,      "nanotime"},
  {GPTLrtc,          utr_rtc,           init_rtc,           "_rtc"},
  {GPTLmpiwtime,     utr_mpiwtime,      init_mpiwtime,      "MPI_Wtime"},
  {GPTLclockgettime, utr_clock_gettime, init_clock_gettime, "clock_gettime"},
  {GPTLpapitime,     utr_papitime,      init_papitime,      "PAPI_get_real_usec"}
};
static const int nfuncentries = sizeof (funclist) / sizeof (Funcentry);

/* 
** The following is for efficiency. Would like to use funclist[funcidx].func
** but compiler complains about non-constant initializer
*/

static int funcidx = 0;               /* default timer is gettimeofday*/  
static double (*ptr2wtimefunc)() = utr_gettimeofday;

#ifdef HAVE_NANOTIME
static float cpumhz = -1.;                        /* init to bad value */
static double cyc2sec = -1;                       /* init to bad value */
static unsigned inline long long nanotime (void); /* read counter (assembler) */
static float get_clockfreq (void);                /* cycles/sec */
#endif

#ifdef UNICOSMP
static double ticks2sec = -1;                     /* init to bad value */
#endif

static const int tablesize = 128*MAX_CHARS;  /* 128 is size of ASCII char set */

/*
** GPTLsetoption: set option value to true or false.
**
** Input arguments:
**   option: option to be set
**   val:    value to which option should be set (nonzero=true, zero=false)
**
** Return value: 0 (success) or GPTLerror (failure)
*/

int GPTLsetoption (const int option,  /* option */
		   const int val)     /* whether to enable */
{
  if (initialized)
    return GPTLerror ("GPTLsetoption: must be called BEFORE GPTLinitialize\n");

  if (option == GPTLabort_on_error) {
    GPTLset_abort_on_error ((bool) val);
    if (verbose)
      printf ("GPTLsetoption: setting abort on error flag to %d\n", val);
    return 0;
  }

  switch (option) {
  case GPTLcpu:
#ifdef HAVE_TIMES
    cpustats.enabled = (bool) val; 
    if (verbose)
      printf ("GPTLsetoption: set cpustats to %d\n", val);
#else
    if (val)
      return GPTLerror ("GPTLsetoption: times() not available\n");
#endif
    return 0;
  case GPTLwall:     
    wallstats.enabled = (bool) val; 
    if (verbose)
      printf ("GPTLsetoption: set wallstats to %d\n", val);
    return 0;
  case GPTLoverhead: 
    overheadstats.enabled = (bool) val; 
    if (verbose)
      printf ("GPTLsetoption: set overheadstats to %d\n", val);
    return 0;
  case GPTLdepthlimit: 
    depthlimit = val; 
    if (verbose)
      printf ("GPTLsetoption: set depthlimit to %d\n", val);
    return 0;
  case GPTLverbose: 
    verbose = (bool) val; 
    if (verbose)
      printf ("GPTLsetoption: set verbose to %d\n", val);
    return 0;
  default:
    break;
  }

#ifdef HAVE_PAPI
  if (GPTL_PAPIsetoption (option, val) == 0) {
    if (val)
      dousepapi = true;
    return 0;
  }
#endif
  return GPTLerror ("GPTLsetoption: option %d not found\n", option);
}

/*
** GPTLsetutr: set underlying timing routine.
**
** Input arguments:
**   option: index which sets function
**
** Return value: 0 (success) or GPTLerror (failure)
*/

int GPTLsetutr (const int option)
{
  int i;

  if (initialized)
    return GPTLerror ("GPTLsetutr: must be called BEFORE GPTLinitialize\n");

  for (i = 0; i < nfuncentries; i++) {
    if (option == (int) funclist[i].option) {
      if (verbose)
	printf ("GPTLsetutr: Setting underlying wallclock timer to %s\n", 
		funclist[i].name);
      funcidx = i;
      return 0;
    }
  }
  return GPTLerror ("GPTLsetutr: unknown option %d\n", option);
}

/*
** GPTLinitialize (): Initialization routine must be called from single-threaded
**   region before any other timing routines may be called.  The need for this
**   routine could be eliminated if not targetting timing library for threaded
**   capability. 
**
** return value: 0 (success) or GPTLerror (failure)
*/

int GPTLinitialize (void)
{
  int i;          /* loop index */
  int t;          /* thread index */

  double t1, t2;  /* returned from underlying timer */

  if (initialized)
    return GPTLerror ("GPTLinitialize: has already been called\n");

  if (threadinit (&nthreads, &maxthreads) < 0)
    return GPTLerror ("GPTLinitialize: bad return from threadinit\n");

  if (get_thread_num (&nthreads, &maxthreads) > 0) 
    return GPTLerror ("GPTLinitialize: must only be called by master thread\n");

  if ((ticks_per_sec = sysconf (_SC_CLK_TCK)) == -1)
    return GPTLerror ("GPTLinitialize: sysconf (_SC_CLK_TCK) failed\n");

  /* Allocate space for global arrays */

  timers        = (Timer **)     GPTLallocate (maxthreads * sizeof (Timer *));
  last          = (Timer **)     GPTLallocate (maxthreads * sizeof (Timer *));
  current_depth = (Nofalse *)    GPTLallocate (maxthreads * sizeof (Nofalse));
  max_depth     = (int *)        GPTLallocate (maxthreads * sizeof (int));
  max_name_len  = (int *)        GPTLallocate (maxthreads * sizeof (int));
  hashtable     = (Hashentry **) GPTLallocate (maxthreads * sizeof (Hashentry *));

  /* Initialize array values */

  for (t = 0; t < maxthreads; t++) {
    timers[t] = 0;
    current_depth[t].depth = 0;
    max_depth[t]     = 0;
    max_name_len[t]  = 0;
    hashtable[t] = (Hashentry *) GPTLallocate (tablesize * sizeof (Hashentry));
    for (i = 0; i < tablesize; i++) {
      hashtable[t][i].nument = 0;
      hashtable[t][i].entries = 0;
    }
  }

#ifdef HAVE_PAPI
  if (GPTL_PAPIinitialize (maxthreads) < 0)
    return GPTLerror ("GPTLinitialize: GPTL_PAPIinitialize failure\n");
#endif

  /* 
  ** start/stop routines sprintf an "unsigned long" into a string for later 
  ** printing. The size of that string is MAX_CHARS (excluding null terminator). 
  ** The following checks that this size is sufficient.
  ** A single byte can hold 2 hex digits.
  */

  if (2*sizeof (void *) > MAX_CHARS) {
    printf ("GPTLinitialize: NOTE: MAX_CHARS may be too small for automatic profiling");
  }

  /* 
  ** Call init routine for underlying timing routine.
  */

  if ((*funclist[funcidx].funcinit)() < 0) {
    printf ("GPTLinitialize: failure initializing %s: reverting underlying timer"
	    " to default %s\n", funclist[funcidx].name, funclist[0].name);
    funcidx = 0;
  }

  ptr2wtimefunc = funclist[funcidx].func;

  t1 = (*ptr2wtimefunc) ();
  t2 = (*ptr2wtimefunc) ();

  if (t1 > t2)
    return GPTLerror ("GPTLinitialize: bad t1=%f t2=%f\n", t1, t2);

  if (verbose) {
    printf ("Per call overhead est. t2-t1=%g should be near zero\n", t2-t1);
    printf ("Underlying wallclock timing routine is %s\n", funclist[funcidx].name);
  }

  initialized = true;
  return 0;
}

/*
** GPTLfinalize (): Finalization routine must be called from single-threaded
**   region. Free all malloc'd space
**
** return value: 0 (success) or GPTLerror (failure)
*/

int GPTLfinalize (void)
{
  int t;                /* thread index */
  Timer *ptr, *ptrnext; /* ll indices */

  if ( ! initialized)
    return GPTLerror ("GPTLfinalize: initialization was not completed\n");

  if (get_thread_num (&nthreads, &maxthreads) > 0) 
    return GPTLerror ("GPTLfinalize: must only be called by master thread\n");

  for (t = 0; t < maxthreads; ++t) {
    free (hashtable[t]);
    for (ptr = timers[t]; ptr; ptr = ptrnext) {
      ptrnext = ptr->next;
      free (ptr);
    }
  }

  free (timers);
  free (current_depth);
  free (max_depth);
  free (max_name_len);
  free (hashtable);

  threadfinalize ();

#ifdef HAVE_PAPI
  GPTL_PAPIfinalize (maxthreads);
#endif

  /* Reset initial values set in GPTLinitialize */

  timers = 0;
  last = 0;
  nthreads = -1;
  maxthreads = -1;
  initialized = false;
  ref_gettimeofday = -1;
  ref_clock_gettime = -1;
  ref_papitime = -1;

  return 0;
}

/*
** GPTLstart: start a timer
**
** Input arguments:
**   name: timer name
**
** Return value: 0 (success) or GPTLerror (failure)
*/

int GPTLstart (const char *name)               /* timer name */
{
  double tp2;      /* time stamp */
  Timer *ptr;      /* linked list pointer */
  Timer **eptr;    /* for realloc */

  int nchars;      /* number of characters in timer */
  int t;           /* thread index (of this thread) */
  int indx;        /* hash table index */
  int nument;      /* number of entries for a hash collision */

#ifdef UNICOSMP
#ifndef SSP
  if (__streaming() == 0) return 0;  /* timers don't work in this situation so disable */
#endif
#endif

  char locname[MAX_CHARS+1]; /* "name" truncated to max allowed number of chars */

  if (disabled)
    return 0;

  if ( ! initialized)
    return GPTLerror ("GPTLstart: GPTLinitialize has not been called\n");

  if ((t = get_thread_num (&nthreads, &maxthreads)) < 0)
    return GPTLerror ("GPTLstart\n");

  /*
  ** If current depth exceeds a user-specified limit for print, just
  ** increment and return
  */

  if (current_depth[t].depth >= depthlimit) {
    ++current_depth[t].depth;
    return 0;
  }

  /* 
  ** Truncate input name if longer than MAX_CHARS characters.
  ** This copy is unnecessary if called from Fortran because the
  ** wrapper code already fixed the input name.
  */

  nchars = MIN (strlen (name), MAX_CHARS);
  strncpy (locname, name, nchars);
  locname[nchars] = '\0';

  /* 
  ** ptr will point to the requested timer in the current list,
  ** or NULL if this is a new entry 
  */

  ptr = getentry (hashtable[t], locname, &indx);

  if (indx >= tablesize)
    return GPTLerror ("GPTLstart: indx=%d must be < tablesize=%d\n", indx, tablesize);

  /* 
  ** Recursion => increment depth in recursion and return.  We need to return 
  ** because we don't want to restart the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */

  if (ptr && ptr->onflg) {
    ++ptr->recurselvl;
    return 0;
  } else {
    ++current_depth[t].depth;
    if (current_depth[t].depth > max_depth[t])
      max_depth[t] = current_depth[t].depth;
  }

  if (ptr) {

    /*
    ** Reset indentation level to ambiguous value if inconsistent with
    ** current value. This will likely happen when the thing being timed is
    ** called from more than 1 branch in the call tree.
    */
  
    if (ptr->depth != current_depth[t].depth)
      ptr->depth = 0;

  } else {

    /* Add a new entry and initialize */

    ptr = (Timer *) GPTLallocate (sizeof (Timer));
    memset (ptr, 0, sizeof (Timer));

    if (nchars > max_name_len[t])
      max_name_len[t] = nchars;

    strcpy (ptr->name, locname);
    ptr->depth = current_depth[t].depth;

    if (timers[t])
      last[t]->next = ptr;
    else
      timers[t] = ptr;

    last[t] = ptr;
    ++hashtable[t][indx].nument;
    nument = hashtable[t][indx].nument;

    eptr = (Timer **) realloc (hashtable[t][indx].entries, nument * sizeof (Timer *));
    if ( ! eptr)
      return GPTLerror ("GPTLstart: realloc error\n");

    hashtable[t][indx].entries           = eptr;
    hashtable[t][indx].entries[nument-1] = ptr;
  }

  ptr->onflg = true;

  /* Get timestamp */
  
  if (cpustats.enabled && get_cpustamp (&ptr->cpu.last_utime, &ptr->cpu.last_stime) < 0)
    return GPTLerror ("GPTLstart: get_cpustamp error");
  
  if (wallstats.enabled) {
    tp2 = (*ptr2wtimefunc) ();
    ptr->wall.last = tp2;
  }

#ifdef HAVE_PAPI
  if (dousepapi && GPTL_PAPIstart (t, &ptr->aux) < 0)
    return GPTLerror ("GPTLstart: error from GPTL_PAPIstart\n");
#endif

  return (0);
}

/*
** GPTLstop: stop a timer
**
** Input arguments:
**   name: timer name
**
** Return value: 0 (success) or -1 (failure)
*/

int GPTLstop (const char *name)               /* timer name */
{
  double tp1;                /* time stamp */
  double delta;              /* diff between 2 time stamps */
  Timer *ptr;                /* linked list pointer */

  int nchars;                /* number of characters in timer */
  int t;                     /* thread number for this process */
  int indx;                  /* index into hash table */

  long usr;                  /* user time (returned from get_cpustamp) */
  long sys;                  /* system time (returned from get_cpustamp) */

#ifdef UNICOSMP
#ifndef SSP
  if (__streaming() == 0) return 0;  /* timers don't work in this situation so disable */
#endif
#endif

  char locname[MAX_CHARS+1]; /* "name" truncated to max allowed number of chars */

  if (disabled)
    return 0;

  /* Get the timestamp */
    
  if (wallstats.enabled) {
    tp1 = (*ptr2wtimefunc) ();
  }

  if (cpustats.enabled && get_cpustamp (&usr, &sys) < 0)
    return GPTLerror (0);

  if ( ! initialized)
    return GPTLerror ("GPTLstop: GPTLinitialize has not been called\n");

  if ((t = get_thread_num (&nthreads, &maxthreads)) < 0)
    return GPTLerror ("GPTLstop\n");

  /*
  ** If current depth exceeds a user-specified limit for print, just
  ** decrement and return
  */

  if (current_depth[t].depth > depthlimit) {
    --current_depth[t].depth;
    return 0;
  }

  /* 
  ** Truncate input name if longer than MAX_CHARS characters.
  ** This copy is unnecessary if called from Fortran because the
  ** wrapper code already fixed the input name.
  */

  nchars = MIN (strlen (name), MAX_CHARS);
  strncpy (locname, name, nchars);
  locname[nchars] = '\0';

  ptr = getentry (hashtable[t], locname, &indx);
  if ( ! ptr) 
    return GPTLerror ("GPTLstop: timer for %s had not been started.\n", locname);

  if ( ! ptr->onflg )
    return GPTLerror ("GPTLstop: timer %s was already off.\n",ptr->name);

  ++ptr->count;

  /* 
  ** Recursion => decrement depth in recursion and return.  We need to return
  ** because we don't want to stop the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */

  if (ptr->recurselvl > 0) {
    ++ptr->nrecurse;
    --ptr->recurselvl;
    return 0;
  } else {
    ptr->onflg = false;
    --current_depth[t].depth;
    if (current_depth[t].depth < 0) {
      current_depth[t].depth = 0;
      return GPTLerror ("GPTLstop: tree depth has become negative.\n");
    }
  }

#ifdef HAVE_PAPI
  if (dousepapi && GPTL_PAPIstop (t, &ptr->aux) < 0)
    return GPTLerror ("GPTLstop: error from GPTL_PAPIstop\n");
#endif

  if (wallstats.enabled) {

    delta = tp1 - ptr->wall.last;
    ptr->wall.accum += delta;

    if (delta < 0.) {
      printf ("GPTLstop: negative delta=%g reset to 0\n", delta);
      delta = 0.;
    }

    if (ptr->count == 1) {
      ptr->wall.max = delta;
      ptr->wall.min = delta;
    } else {
      if (delta > ptr->wall.max)
	ptr->wall.max = delta;
      if (delta < ptr->wall.min)
	ptr->wall.min = delta;
    }
  }

  if (cpustats.enabled) {
    ptr->cpu.accum_utime += usr - ptr->cpu.last_utime;
    ptr->cpu.accum_stime += sys - ptr->cpu.last_stime;
    ptr->cpu.last_utime   = usr;
    ptr->cpu.last_stime   = sys;
  }
  return 0;
}

/*
** GPTLenable: enable timers
**
** Return value: 0 (success)
*/

int GPTLenable (void)
{
  disabled = false;
  return (0);
}

/*
** GPTLdisable: disable timers
**
** Return value: 0 (success)
*/

int GPTLdisable (void)
{
  disabled = true;
  return (0);
}

/*
** GPTLstamp: Compute timestamp of usr, sys, and wallclock time (seconds)
**
** Output arguments:
**   wall: wallclock
**   usr:  user time
**   sys:  system time
**
** Return value: 0 (success) or GPTLerror (failure)
*/

int GPTLstamp (double *wall, double *usr, double *sys)
{
  struct timeval tp;         /* argument to gettimeofday */
  struct tms buf;            /* argument to times */

  if ( ! initialized)
    return GPTLerror ("GPTLstamp: GPTLinitialize has not been called\n");

#ifdef HAVE_TIMES
  *usr = 0;
  *sys = 0;

  if (times (&buf) == -1)
    return GPTLerror ("GPTLstamp: times() failed. Results bogus\n");

  *usr = buf.tms_utime / (double) ticks_per_sec;
  *sys = buf.tms_stime / (double) ticks_per_sec;

  gettimeofday (&tp, 0);
  *wall = tp.tv_sec + 1.e-6*tp.tv_usec;
  return 0;
#else
  return GPTLerror ("GPTLstamp: times() not available\n");
#endif
}

/*
** GPTLreset: reset all known timers to 0
**
** Return value: 0 (success) or GPTLerror (failure)
*/

int GPTLreset (void)
{
  int t;             /* index over threads */
  Timer *ptr;        /* linked list index */

  if ( ! initialized)
    return GPTLerror ("GPTLreset: GPTLinitialize has not been called\n");

  /* Only allow the master thread to reset timers */

  if (get_thread_num (&nthreads, &maxthreads) > 0) 
    return GPTLerror ("GPTLreset: must only be called by master thread\n");

  for (t = 0; t < nthreads; t++) {
    for (ptr = timers[t]; ptr; ptr = ptr->next) {
      ptr->onflg = false;
      ptr->count = 0;
      memset (&ptr->wall, 0, sizeof (ptr->wall));
      memset (&ptr->cpu, 0, sizeof (ptr->cpu));
#ifdef HAVE_PAPI
      memset (&ptr->aux, 0, sizeof (ptr->aux));
#endif
    }
  }

  if (verbose)
    printf ("GPTLreset: accumulators for all timers set to zero\n");

  return 0;
}

/* 
** GPTLpr: Print values of all timers
**
** Input arguments:
**   id: integer to append to string "timing."
**
** Return value: 0 (success) or GPTLerror (failure)
*/

int GPTLpr (const int id)   /* output file will be named "timing.<id>" */
{
  FILE *fp;                 /* file handle to write to */
  Timer *ptr;               /* walk through master thread linked list */
  Timer *tptr;              /* walk through slave threads linked lists */
  Timer sumstats;           /* sum of same timer stats over threads */
  int i, ii, n, t;          /* indices */
  unsigned long totcount;   /* total timer invocations */
  unsigned long totrecurse; /* total recursive timer invocations */
  char outfile[12];         /* name of output file: timing.xxxx */
  float *sum;               /* sum of overhead values (per thread) */
  float osum;               /* sum of overhead over threads */
  double utr_overhead;      /* overhead of calling underlying timing routine */
  bool found;               /* jump out of loop when name found */
  bool foundany;            /* whether summation print necessary */
  bool first;               /* flag 1st time entry found */

  if ( ! initialized)
    return GPTLerror ("GPTLpr: GPTLinitialize() has not been called\n");

  if (get_thread_num (&nthreads, &maxthreads) > 0) 
    return GPTLerror ("GPTLpr: must only be called by master thread\n");

  if (id < 0 || id > 9999)
    return GPTLerror ("GPTLpr: bad id=%d for output file. Must be >= 0 and < 10000\n", id);

  sprintf (outfile, "timing.%d", id);

  if ( ! (fp = fopen (outfile, "w")))
    fp = stderr;

#ifdef HAVE_NANOTIME
  fprintf (fp, "Clock rate = %f MHz\n", cpumhz);
#endif

#ifdef HAVE_PAPI
  if (dousepapi) {
    if (GPTL_PAPIis_multiplexed ())
      fprintf (fp, "PAPI event multiplexing was ON\n");
    else
      fprintf (fp, "PAPI event multiplexing was OFF\n");
    GPTL_PAPIprintenabled (fp);
  }
#endif

  /*
  ** Estimate underlying timing routine overhead
  */

  utr_overhead = utr_getoverhead ();
  fprintf (fp, "Underlying timing routine was %s\n", funclist[funcidx].name);
  fprintf (fp, "Per-call utr overhead est: %g sec\n\n", utr_overhead);
  fprintf (fp, "If overhead stats are printed, roughly half the estimated number is\n");
  fprintf (fp, "embedded in the wallclock (and/or PAPI counter) stats for each timer\n\n");

  sum = (float *) GPTLallocate (nthreads * sizeof (float));
  
  for (t = 0; t < nthreads; ++t) {
    if (t > 0)
      fprintf (fp, "\n");
    fprintf (fp, "Stats for thread %d:\n", t);

    for (n = 0; n < max_depth[t]; ++n)    /* max indent level (depth starts at 1) */
      fprintf (fp, "  ");
    for (n = 0; n < max_name_len[t]; ++n) /* longest timer name */
      fprintf (fp, " ");

    fprintf (fp, "Called Recurse ");

    /* Print strings for enabled timer types */

    if (cpustats.enabled)
      fprintf (fp, "%s", cpustats.str);
    if (wallstats.enabled) {
      fprintf (fp, "%s", wallstats.str);
      if (overheadstats.enabled)
	fprintf (fp, "%s", overheadstats.str);
    }

#ifdef HAVE_PAPI
    GPTL_PAPIprstr (fp, overheadstats.enabled);
#endif

    fprintf (fp, "\n");        /* Done with titles, go to next line */

    for (ptr = timers[t]; ptr; ptr = ptr->next)
      printstats (ptr, fp, t, true, utr_overhead);

    /* 
    ** Sum of overhead across timers is meaningful.
    ** Factor of 2 is because there are 2 utr calls per start/stop pair.
    */

    sum[t]     = 0;
    totcount   = 0;
    totrecurse = 0;
    for (ptr = timers[t]; ptr; ptr = ptr->next) {
      sum[t]     += ptr->count * 2 * utr_overhead;
      totcount   += ptr->count;
      totrecurse += ptr->nrecurse;
    }
    if (wallstats.enabled && overheadstats.enabled)
      fprintf (fp, "Overhead sum          = %9.3f wallclock seconds\n", sum[t]);
    fprintf (fp, "Total calls           = %lu\n", totcount);
    fprintf (fp, "Total recursive calls = %lu\n", totrecurse);
  }

  /* Print per-name stats for all threads */

  if (nthreads > 1) {
    fprintf (fp, "\nSame stats sorted by timer for threaded regions:\n");
    fprintf (fp, "Thd ");

    for (n = 0; n < max_name_len[0]; ++n) /* longest timer name */
      fprintf (fp, " ");

    fprintf (fp, "Called Recurse ");

    if (cpustats.enabled)
      fprintf (fp, "%s", cpustats.str);
    if (wallstats.enabled) {
      fprintf (fp, "%s", wallstats.str);
      if (overheadstats.enabled)
	fprintf (fp, "%s", overheadstats.str);
    }

#ifdef HAVE_PAPI
    GPTL_PAPIprstr (fp, overheadstats.enabled);
#endif

    fprintf (fp, "\n");

    for (ptr = timers[0]; ptr; ptr = ptr->next) {
      
      /* 
      ** To print sum stats, first create a new timer then copy thread 0
      ** stats into it. then sum using "add", and finally print.
      */

      foundany = false;
      first = true;
      sumstats = *ptr;
      for (t = 1; t < nthreads; ++t) {
	found = false;
	for (tptr = timers[t]; tptr && ! found; tptr = tptr->next) {
	  if (STRMATCH (ptr->name, tptr->name)) {

	    /* Only print thread 0 when this timer found for other threads */

	    if (first) {
	      first = false;
	      fprintf (fp, "%3.3d ", 0);
	      printstats (ptr, fp, 0, false, utr_overhead);
	    }

	    found = true;
	    foundany = true;
	    fprintf (fp, "%3.3d ", t);
	    printstats (tptr, fp, 0, false, utr_overhead);
	    add (&sumstats, tptr);
	  }
	}
      }

      if (foundany) {
	fprintf (fp, "SUM ");
	printstats (&sumstats, fp, 0, false, utr_overhead);
	fprintf (fp, "\n");
      }
    }

    /* Repeat overhead print in loop over threads */

    if (wallstats.enabled && overheadstats.enabled) {
      osum = 0.;
      for (t = 0; t < nthreads; ++t) {
	fprintf (fp, "OVERHEAD.%3.3d (wallclock seconds) = %9.3f\n", t, sum[t]);
	osum += sum[t];
      }
      fprintf (fp, "OVERHEAD.SUM (wallclock seconds) = %9.3f\n", osum);
    }
  }

  /* Print hash table stats */

  for (t = 0; t < nthreads; t++) {
    first = true;
    for (i = 0; i < tablesize; i++) {
      int nument = hashtable[t][i].nument;
      if (nument > 1) {
	if (first) {
	  first = false;
	  fprintf (fp, "\nthread %d had some hash collisions:\n", t);
	}
	fprintf (fp, "hashtable[%d][%d] had %d entries:", t, i, nument);
	for (ii = 0; ii < nument; ii++)
	  fprintf (fp, " %s", hashtable[t][i].entries[ii]->name);
	fprintf (fp, "\n");
      }
    }
  }

  free (sum);
  return 0;
}

/* 
** printstats: print a single timer
**
** Input arguments:
**   timer:    timer for which to print stats
**   fp:       file descriptor to write to
**   t:        thread number
**   doindent: whether to indent
*/

static void printstats (const Timer *timer,     /* timer to print */
			FILE *fp,               /* file descriptor to write to */
			const int t,            /* thread number */
			const bool doindent,    /* whether indenting will be done */
			double utr_overhead)    /* underlying timing routine overhead */
{
  int i;               /* index */
  int indent;          /* index for indenting */
  int extraspace;      /* for padding to length of longest name */
  long ticks_per_sec;  /* returned from sysconf */
  float usr;           /* user time */
  float sys;           /* system time */
  float usrsys;        /* usr + sys */
  float elapse;        /* elapsed time */
  float wallmax;       /* max wall time */
  float wallmin;       /* min wall time */

  if ((ticks_per_sec = sysconf (_SC_CLK_TCK)) == -1)
    (void) GPTLerror ("printstats: token _SC_CLK_TCK is not defined\n");

  if (timer->onflg)
    fprintf (stderr, "GPTLpr: timer %s had not been turned off\n", timer->name);

  /* Indent to depth of this timer */

  if (doindent)
    for (indent = 0; indent < timer->depth; ++indent)  /* depth starts at 1 */
      fprintf (fp, "  ");

  fprintf (fp, "%s", timer->name);

  /* Pad to length of longest name */

  extraspace = max_name_len[t] - strlen (timer->name);
  for (i = 0; i < extraspace; ++i)
    fprintf (fp, " ");

  /* Pad to max indent level */

  if (doindent)
    for (indent = timer->depth; indent < max_depth[t]; ++indent)
      fprintf (fp, "  ");

  if (timer->nrecurse > 0)
    fprintf (fp, "%8ld %5ld ", timer->count, timer->nrecurse);
  else
    fprintf (fp, "%8ld   -   ", timer->count);

  if (cpustats.enabled) {
    usr = timer->cpu.accum_utime / (float) ticks_per_sec;
    sys = timer->cpu.accum_stime / (float) ticks_per_sec;
    usrsys = usr + sys;
    fprintf (fp, "%9.3f %9.3f %9.3f ", usr, sys, usrsys);
  }

  if (wallstats.enabled) {
    elapse = timer->wall.accum;
    wallmax = timer->wall.max;
    wallmin = timer->wall.min;
    fprintf (fp, "%9.3f %9.3f %9.3f ", elapse, wallmax, wallmin);

    /*
    ** Factor of 2 is because there are 2 utr calls per start/stop pair.
    */

    if (overheadstats.enabled) {
      fprintf (fp, "%13.3f ", timer->count * 2 * utr_overhead);
    }
  }

#ifdef HAVE_PAPI
  GPTL_PAPIpr (fp, &timer->aux, t, timer->count, overheadstats.enabled);
#endif

  fprintf (fp, "\n");
}

/* 
** add: add the contents of tin to tout
**
** Input arguments:
**   tin:  input timer
** Input/output arguments:
**   tout: output timer summed into
*/

static void add (Timer *tout,   
		 const Timer *tin)
{
  if (wallstats.enabled) {
    tout->count      += tin->count;
    tout->wall.accum += tin->wall.accum;
    
    tout->wall.max = MAX (tout->wall.max, tin->wall.max);
    tout->wall.min = MIN (tout->wall.min, tin->wall.min);
  }

  if (cpustats.enabled) {
    tout->cpu.accum_utime += tin->cpu.accum_utime;
    tout->cpu.accum_stime += tin->cpu.accum_stime;
  }
#ifdef HAVE_PAPI
  GPTL_PAPIadd (&tout->aux, &tin->aux);
#endif
}

/*
** get_cpustamp: Invoke the proper system timer and return stats.
**
** Output arguments:
**   usr: user time
**   sys: system time
**
** Return value: 0 (success)
*/

static inline int get_cpustamp (long *usr, long *sys)
{
#ifdef HAVE_TIMES
  struct tms buf;

  (void) times (&buf);
  *usr = buf.tms_utime;
  *sys = buf.tms_stime;
  return 0;
#else
  return GPTLerror ("get_cpustamp: times() not available\n");
#endif
}

/*
** GPTLquery: return current status info about a timer. If certain stats are not 
** enabled, they should just have zeros in them. If PAPI is not enabled, input
** counter info is ignored.
** 
** Input args:
**   name:        timer name
**   maxcounters: max number of PAPI counters to get info for
**
** Output args:
**   t:                  thread number (if < 0, it will be OUTPUT as the actual thread number)
**   count:              number of times this timer was called
**   onflg:              whether timer is currently on
**   wallclock:          accumulated wallclock time
**   usr:                accumulated user CPU time
**   sys:                accumulated system CPU time
**   papicounters_accum: accumulated PAPI counters
*/

int GPTLquery (const char *name, 
	       int t,
	       int *count,
	       int *onflg,
	       double *wallclock,
	       double *usr,
	       double *sys,
	       long *papicounters_out,
	       const int maxcounters)
{
  Timer *ptr;                /* linked list pointer */
  int nchars;                /* number of characters in timer */
  int indx;
  char locname[MAX_CHARS+1]; /* "name" truncated to max allowed number of chars */
  
  if ( ! initialized)
    return GPTLerror ("GPTLquery: GPTLinitialize has not been called\n");
  
/*
** If t is < 0, assume the request is for the current thread
*/
  
  if (t < 0) {
    if ((t = get_thread_num (&nthreads, &maxthreads)) < 0)
      return GPTLerror ("GPTLquery: get_thread_num failure\n");
  } else {
    if (t >= maxthreads)
      return GPTLerror ("GPTLquery: requested thread %d is too big\n", t);
  }
  
  /* Truncate input name if longer than MAX_CHARS characters  */

  nchars = MIN (strlen (name), MAX_CHARS);
  strncpy (locname, name, nchars);
  locname[nchars] = '\0';

  ptr = getentry (hashtable[t], locname, &indx);
  if ( !ptr)
    return GPTLerror ("GPTLquery: requested timer %s does not exist\n", locname);

  if (indx >= tablesize)
    return GPTLerror ("GPTLstart: indx=%d must be < tablesize=%d\n", indx, tablesize);

  *onflg     = ptr->onflg;
  *count     = ptr->count;
  *wallclock = ptr->wall.accum;
  *usr       = ptr->cpu.accum_utime;
  *sys       = ptr->cpu.accum_stime;
#ifdef HAVE_PAPI
  GPTL_PAPIquery (&ptr->aux, papicounters_out, maxcounters);
#endif
  return 0;
}

/*
** getentry: find the entry in the hash table and return a pointer to it.
**
** Input args:
**   hashtable: the hashtable (array)
**   name:      string to be hashed on (specifically, summed)
** Output args:
**   indx:      hashtable index
**
** Return value: pointer to the entry, or NULL if not found
*/

static inline Timer *getentry (const Hashentry *hashtable, /* hash table */
			       const char *name,           /* name to hash */
			       int *indx)                  /* hash index */
{
  int i;                 /* loop index */
  const char *c = name;  /* pointer to elements of "name" */

  /* Generate the hash value by summing values of the chars in "name" */

  for (*indx = 0; *c; c++)
    *indx += *c;

  /* 
  ** If nument exceeds 1 there was a hash collision and we must search
  ** linearly through an array for a match
  */

  for (i = 0; i < hashtable[*indx].nument; i++)
    if (STRMATCH (name, hashtable[*indx].entries[i]->name))
      return hashtable[*indx].entries[i];

  return 0;
}

/*
** These routines were moved from threadutil.c to here only to allow inlining.
*/

#if ( defined THREADED_OMP )
#include <omp.h>

/*
** get_thread_num: determine thread number of the calling thread
**
** Input args:
**   nthreads:   number of threads
**   maxthreads: number of threads (unused in OpenMP case)
**
** Return value: thread number (success) or GPTLerror (failure)
*/

static inline int get_thread_num (int *nthreads, int *maxthreads)
{
  int t;       /* thread number */

  if ((t = omp_get_thread_num ()) >= *nthreads)
    return GPTLerror ("get_thread_num: returned id %d exceed numthreads %d\n",
		      t, *nthreads);

  return t;
}

#elif ( ! defined THREADED_PTHREADS )

static inline int get_thread_num (int *nthreads, int *maxthreads)
{
  return 0;
}

#endif

/*
** Add entry points for when -finstrument-functions was set on gcc compile
** line.
*/

#ifdef __cplusplus
extern "C" {
#endif

void __cyg_profile_func_enter (void *this_fn,
                               void *call_site)
{
  /* 64 is big enough to hold a 128-bit address */

  char locname[64+1];
  sprintf (locname, "%lx", (unsigned long) this_fn);
  (void) GPTLstart (locname);
}

void __cyg_profile_func_exit (void *this_fn,
                              void *call_site)
{
  /* 64 is big enough to hold a 128-bit address */

  char locname[64+1];
  sprintf (locname, "%lx", (unsigned long) this_fn);
  (void) GPTLstop (locname);
}

#ifdef __cplusplus
};
#endif

#ifdef HAVE_NANOTIME
#ifdef BIT64
/* 64-bit code copied from PAPI library */
static inline unsigned long long nanotime (void)
{
    unsigned long long val;
    do {
      unsigned int a,d;
      asm volatile("rdtsc" : "=a" (a), "=d" (d));
      (val) = ((unsigned long)a) | (((unsigned long)d)<<32);
    } while(0);

    return (val);
}
#else
static inline unsigned long long nanotime (void)
{
  unsigned long long val;
  __asm__ __volatile__("rdtsc" : "=A" (val) : );
  return (val);
}
#endif

#define LEN 4096

static float get_clockfreq ()
{
  FILE *fd = 0;
  char buf[LEN];
  int is;

  if ( ! (fd = fopen ("/proc/cpuinfo", "r"))) {
    printf ("get_clockfreq: can't open /proc/cpuinfo\n");
    return -1.;
  }

  while (fgets (buf, LEN, fd)) {
    if (strncmp (buf, "cpu MHz", 7) == 0) {
      for (is = 7; buf[is] != '\0' && !isdigit (buf[is]); is++);
      if (isdigit (buf[is]))
	return atof (&buf[is]);
    }
  }

  return -1.;
}
#endif

/*
** The following are the set of underlying timing routines which may or may
** not be available. And their accompanying init routines.
** NANOTIME is currently only available on x86.
*/

static int init_nanotime ()
{
#ifdef HAVE_NANOTIME
  if ((cpumhz = get_clockfreq ()) < 0)
    return GPTLerror ("Can't get clock freq\n");

  if (verbose)
    printf ("init_nanotime: Clock rate = %f MHz\n", cpumhz);
  cyc2sec = 1./(cpumhz * 1.e6);
  return 0;
#else
  return GPTLerror ("init_nanotime: not enabled\n");
#endif
}

static inline double utr_nanotime ()
{
#ifdef HAVE_NANOTIME
  double timestamp;
  timestamp = nanotime () * cyc2sec;
  return timestamp;
#else
  (void) GPTLerror ("utr_nanotime: not enabled\n");
  return -1.;
#endif
}

/*
** rtc is currently only available on UNICOSMP
*/

static int init_rtc ()
{
#ifdef UNICOSMP
  extern long long rtc_rate_();
  ticks2sec = 1./rtc_rate_();
  if (verbose)
    printf ("init_rtc: ticks per sec=%g\n", rtc_rate_();
  return 0;
#else
  return GPTLerror ("init_rtc: not enabled\n");
#endif
}
  
static inline double utr_rtc ()
{
#ifdef UNICOSMP
  return _rtc () * ticks2sec;
#else
  (void) GPTLerror ("utr_rtc: not enabled\n");
  return -1.;
#endif
}

/*
** MPI_Wtime requires the MPI lib.
*/

static int init_mpiwtime ()
{
#if ( defined HAVE_LIBMPI ) || ( defined HAVE_LIBMPICH )
  return 0;
#else
  return GPTLerror ("utr_mpiwtime: not enabled\n");
#endif
}

static inline double utr_mpiwtime ()
{
#if ( defined HAVE_LIBMPI ) || ( defined HAVE_LIBMPICH )
  return MPI_Wtime ();
#else
  (void) GPTLerror ("utr_mpiwtime: not enabled\n");
  return -1.;
#endif
}

/*
** PAPI_get_real_usec requires the PAPI lib.
*/

static int init_papitime ()
{
#ifdef HAVE_PAPI
  ref_papitime = PAPI_get_real_usec ();
  if (verbose)
    printf ("init_papitime: ref_papitime=%ld\n", (long) ref_papitime);
  return 0;
#else
  return GPTLerror ("init_papitime: not enabled\n");
#endif
}
  
static inline double utr_papitime ()
{
#ifdef HAVE_PAPI
  return (PAPI_get_real_usec () - ref_papitime) * 1.e-6;
#else
  (void) GPTLerror ("utr_papitime: not enabled\n");
  return -1.;
#endif
}

/* 
** Probably need to load with -lrt for this one to work 
*/

static int init_clock_gettime ()
{
#ifdef HAVE_LIBRT
  struct timespec tp;
  (void) clock_gettime (CLOCK_REALTIME, &tp);
  ref_clock_gettime = tp.tv_sec;
  if (verbose)
    printf ("init_clock_gettime: ref_clock_gettime=%ld\n", (long) ref_clock_gettime);
  return 0;
#else
  return GPTLerror ("init_clock_gettime: not enabled\n");
#endif
}

static inline double utr_clock_gettime ()
{
#ifdef HAVE_LIBRT
  struct timespec tp;
  (void) clock_gettime (CLOCK_REALTIME, &tp);
  return (tp.tv_sec - ref_clock_gettime) + 1.e-9*tp.tv_nsec;
#else
  (void) GPTLerror ("utr_clock_gettime: not enabled\n");
  return -1.;
#endif
}

/*
** Default available most places: gettimeofday
*/

static int init_gettimeofday ()
{
#ifdef HAVE_GETTIMEOFDAY
  struct timeval tp;
  (void) gettimeofday (&tp, 0);
  ref_gettimeofday = tp.tv_sec;
  if (verbose)
    printf ("init_gettimeofday: ref_gettimeofday=%ld\n", (long) ref_gettimeofday);
  return 0;
#else
  return GPTLerror ("init_gettimeofday: not enabled\n");
#endif
}

static inline double utr_gettimeofday ()
{
#ifdef HAVE_GETTIMEOFDAY
  struct timeval tp;
  (void) gettimeofday (&tp, 0);
  return (tp.tv_sec - ref_gettimeofday) + 1.e-6*tp.tv_usec;
#else
  return GPTLerror ("utr_gettimeofday: not enabled\n");
#endif
}

/* 
** Determine underlying timing routine overhead: call it 100 times.
*/

static double utr_getoverhead ()
{
  double val1;
  double val2;
  int i;

  val1 = (*ptr2wtimefunc)();
  for (i = 0; i < 10; ++i) {
    val2 = (*ptr2wtimefunc)();
    val2 = (*ptr2wtimefunc)();
    val2 = (*ptr2wtimefunc)();
    val2 = (*ptr2wtimefunc)();
    val2 = (*ptr2wtimefunc)();
    val2 = (*ptr2wtimefunc)();
    val2 = (*ptr2wtimefunc)();
    val2 = (*ptr2wtimefunc)();
    val2 = (*ptr2wtimefunc)();
    val2 = (*ptr2wtimefunc)();
  }
  return 0.01 * (val2 - val1);
}
