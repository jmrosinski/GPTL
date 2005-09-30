#include <stdlib.h>        /* malloc */
#include <sys/time.h>      /* gettimeofday */
#include <sys/times.h>     /* times */
#include <unistd.h>        /* gettimeofday */
#include <stdio.h>
#include <string.h>        /* memset, strcmp (via STRMATCH) */
#include <ctype.h>         /* isdigit */
#include <assert.h>

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
static bool initialized = false; /* GPTLinitialize has been called */

typedef struct {
  const Option option;           /* wall, cpu, etc. */
  const char *str;               /* descriptive string for printing */
  bool enabled;                  /* flag */
} Settings;

/* Options, print strings, and default enable flags */

static Settings cpustats =      {GPTLcpu,      "Usr       sys       usr+sys   ", false};
static Settings wallstats =     {GPTLwall,     "Wallclock max       min       ", true };
static Settings overheadstats = {GPTLoverhead, "Overhead  utr est.  "          , true };

static Hashentry **hashtable;    /* table of entries */
static long ticks_per_sec;       /* clock ticks per second */

/* Local function prototypes */

static void printstats (const Timer *, FILE *, const int, const bool, float);
static void add (Timer *, const Timer *);
static inline int get_cpustamp (long *, long *);

#ifdef NANOTIME
static float cpumhz = -1.;                        /* init to bad value */
static unsigned inline long long nanotime (void); /* read counter (assembler) */
static float get_clockfreq (void);                /* cycles/sec */
#endif

/* functions relating to the Underlying Timing Routine (e.g. gettimeofday) */

static inline void utr_get (UTRtype *);
static inline void utr_save (UTRtype *, const UTRtype *tpin);
static inline void utr_get_delta (const UTRtype *, const UTRtype *, UTRtype *);
static inline float utr_tofloat (const UTRtype *);
static inline void utr_sum (UTRtype *, const UTRtype *);
static float utr_getoverhead (void);
static float utr_norm (float);

#ifdef NUMERIC_TIMERS
static const int tablesize = 16*16*16;       /* 3 hex digits of input name */
static inline Timer *getentry (const Hashentry *, const unsigned long, int *);
#else
static const int tablesize = 128*MAX_CHARS;  /* 128 is size of ASCII char set */
static inline Timer *getentry (const Hashentry *, const char *, int *);
#endif

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

#ifdef DISABLE_TIMERS
  return 0;
#endif

  if (initialized)
    return GPTLerror ("GPTLsetoption: must be called BEFORE GPTLinitialize\n");

  if (option == GPTLabort_on_error) {
    GPTLset_abort_on_error ((bool) val);
    printf ("GPTLsetoption: setting abort on error flag to %d\n", val);
    return 0;
  }

  switch (option) {
  case GPTLcpu:      
    cpustats.enabled = (bool) val; 
    printf ("GPTLsetoption: set cpustats to %d\n", val);
    return 0;
  case GPTLwall:     
    wallstats.enabled = (bool) val; 
    printf ("GPTLsetoption: set wallstats to %d\n", val);
    return 0;
  case GPTLoverhead: 
    overheadstats.enabled = (bool) val; 
    printf ("GPTLsetoption: set overheadstats to %d\n", val);
    return 0;
  default:
    break;
  }

#ifdef HAVE_PAPI
  if (GPTL_PAPIsetoption (option, val) == 0)
    return 0;
#endif
  return GPTLerror ("GPTLsetoption: option %d not found\n", option);
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

#ifdef DISABLE_TIMERS
  return 0;
#endif

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

#ifdef NUMERIC_TIMERS
  /* 
  ** start/stop routines sprintf an "unsigned long" into a string for later 
  ** printing. The size of that string is MAX_CHARS (excluding null terminator). 
  ** The following assert() ensures that this size is sufficient.
  ** A single byte can hold 2 hex digits.
  */

  assert (MAX_CHARS >= 2*sizeof (long));
#endif

#ifdef NANOTIME
  if ((cpumhz = get_clockfreq ()) < 0)
    return GPTLerror ("Can't get clock freq\n");
#endif

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

#ifdef DISABLE_TIMERS
  return 0;
#endif

  if ( ! initialized)
    return GPTLerror ("GPTLfinalize: GPTLinitialize() has not been called\n");

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
  initialized = false;
  return 0;
}

/*
** GPTLstart: start a timer
**
** Input arguments:
**   name: timer name OR
**   tag:  number (e.g. maybe an address)
**
** Return value: 0 (success) or GPTLerror (failure)
*/

#ifdef NUMERIC_TIMERS
inline int GPTLstart (const unsigned long tag) /* timer tag */
#else
int GPTLstart (const char *name)               /* timer name */
#endif
{
  UTRtype tp1, tp2;          /* argument returned from underlying timing routine */
  UTRtype delta;             /* diff between 2 UTRtypes */
  Timer *ptr;                /* linked list pointer */
  Timer **eptr;              /* for realloc */

  int nchars;                /* number of characters in timer */
  int t;                     /* thread index (of this thread) */
  int indx;                  /* hash table index */
  int nument;                /* number of entries for a hash collision */

  char locname[MAX_CHARS+1]; /* "name" truncated to max allowed number of chars */

#ifdef DISABLE_TIMERS
  return 0;
#endif

  if ((t = get_thread_num (&nthreads, &maxthreads)) < 0)
    return GPTLerror ("GPTLstart\n");

  /* 
  ** 1st calls to overheadstart and gettimeofday are solely for overhead timing
  ** Would prefer to start overhead calcs before get_thread_num, but the
  ** thread number is required by PAPI counters
  */

  if (overheadstats.enabled) {
#ifdef HAVE_PAPI
    (void) GPTL_PAPIoverheadstart (t);
#endif
    if (wallstats.enabled)
      utr_get (&tp1);
  }

  if ( ! initialized)
    return GPTLerror ("GPTLstart: GPTLinitialize has not been called\n");

#ifdef NUMERIC_TIMERS
  ptr = getentry (hashtable[t], tag, &indx);
#else

  /* Truncate input name if longer than MAX_CHARS characters  */

  nchars = MIN (strlen (name), MAX_CHARS);
  strncpy (locname, name, nchars);
  locname[nchars] = '\0';

  /* 
  ** ptr will point to the requested timer in the current list,
  ** or NULL if this is a new entry 
  */

  ptr = getentry (hashtable[t], locname, &indx);
#endif

  assert (indx < tablesize);

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

#ifdef NUMERIC_TIMERS

    /* 
    ** Convert tag to a string for printing.
    ** nchars is guaranteed by assert in init. to be <= MAX_CHARS
    */

    sprintf (locname, "%lx", tag);
    nchars = strlen (locname);
    ptr->tag = tag;
#endif

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

  if (cpustats.enabled && get_cpustamp (&ptr->cpu.last_utime, &ptr->cpu.last_stime) < 0)
    return GPTLerror ("GPTLstart: get_cpustamp error");
  
  /*
  ** The 2nd system timer call is used both for overhead estimation and
  ** the input timer
  */
  
  if (wallstats.enabled) {
    utr_get (&tp2);
    utr_save (&ptr->wall.last, &tp2);
    if (overheadstats.enabled) {
      utr_get_delta (&tp1, &tp2, &delta);
      ptr->wall.overhead += utr_tofloat (&delta);
    }
  }

#ifdef HAVE_PAPI
  if (GPTL_PAPIstart (t, &ptr->aux) < 0)
    return GPTLerror ("GPTLstart: error from GPTL_PAPIstart\n");

  if (overheadstats.enabled)
    (void) GPTL_PAPIoverheadstop (t, &ptr->aux);
#endif

  return (0);
}

/*
** GPTLstop: stop a timer
**
** Input arguments:
**   name: timer name OR
**   tag:  number (e.g. maybe an address)
**
** Return value: 0 (success) or -1 (failure)
*/

#ifdef NUMERIC_TIMERS
inline int GPTLstop (const unsigned long tag) /* timer tag */
#else
int GPTLstop (const char *name)               /* timer name */
#endif
{
  float delta_wtime;         /* floating point wallclock change */
  UTRtype tp1, tp2;          /* argument to gettimeofday() */
  UTRtype delta;             /* diff between 2 UTRtypes */
  Timer *ptr;                /* linked list pointer */

  int nchars;                /* number of characters in timer */
  int t;                     /* thread number for this process */
  int indx;                  /* index into hash table */

  long usr;                  /* user time (returned from get_cpustamp) */
  long sys;                  /* system time (returned from get_cpustamp) */

  char locname[MAX_CHARS+1]; /* "name" truncated to max allowed number of chars */

#ifdef DISABLE_TIMERS
  return 0;
#endif

  if ((t = get_thread_num (&nthreads, &maxthreads)) < 0)
    return GPTLerror ("GPTLstop\n");

#ifdef HAVE_PAPI
  if (overheadstats.enabled)
    (void) GPTL_PAPIoverheadstart (t);
#endif

  /*
  ** The 1st system timer call is used both for overhead estimation and
  ** the input timer
  */
    
  if (wallstats.enabled)
    utr_get (&tp1);

  if (cpustats.enabled && get_cpustamp (&usr, &sys) < 0)
    return GPTLerror (0);

  if ( ! initialized)
    return GPTLerror ("GPTLstop: GPTLinitialize has not been called\n");

#ifdef NUMERIC_TIMERS
  ptr = getentry (hashtable[t], tag, &indx);
  if ( ! ptr) 
    return GPTLerror ("GPTLstop: timer for %lx had not been started.\n", tag);
#else
  nchars = MIN (strlen (name), MAX_CHARS);
  strncpy (locname, name, nchars);
  locname[nchars] = '\0';

  ptr = getentry (hashtable[t], locname, &indx);
  if ( ! ptr) 
    return GPTLerror ("GPTLstop: timer for %s had not been started.\n", locname);
#endif

  if ( ! ptr->onflg )
    return GPTLerror ("GPTLstop: timer %s was already off.\n",ptr->name);

#ifdef HAVE_PAPI
  if (GPTL_PAPIstop (t, &ptr->aux) < 0)
    return GPTLerror ("GPTLstop: error from GPTL_PAPIstop\n");
#endif

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

  if (wallstats.enabled) {

    utr_get_delta (&ptr->wall.last, &tp1, &delta);
    delta_wtime = utr_tofloat (&delta);
    utr_sum (&ptr->wall.accum, &delta);

    if (ptr->count == 1) {
      ptr->wall.max = delta_wtime;
      ptr->wall.min = delta_wtime;
    } else {
      if (delta_wtime > ptr->wall.max)
	ptr->wall.max = delta_wtime;
      if (delta_wtime < ptr->wall.min)
	ptr->wall.min = delta_wtime;
    }

    /* 2nd system timer call is solely for overhead timing */

    if (overheadstats.enabled) {
      utr_get (&tp2);
      utr_get_delta (&tp1, &tp2, &delta);
      ptr->wall.overhead += utr_tofloat (&delta);
    }
  }

  if (cpustats.enabled) {
    ptr->cpu.accum_utime += usr - ptr->cpu.last_utime;
    ptr->cpu.accum_stime += sys - ptr->cpu.last_stime;
    ptr->cpu.last_utime   = usr;
    ptr->cpu.last_stime   = sys;
  }

#ifdef HAVE_PAPI
  if (overheadstats.enabled)
    (void) GPTL_PAPIoverheadstop (t, &ptr->aux);
#endif

  return 0;
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

#ifdef DISABLE_TIMERS
  return 0;
#endif

  if ( ! initialized)
    return GPTLerror ("GPTLstamp: GPTLinitialize has not been called\n");

  *usr = 0;
  *sys = 0;

  if (times (&buf) == -1)
    return GPTLerror ("GPTLstamp: times() failed. Results bogus\n");

  *usr = buf.tms_utime / (double) ticks_per_sec;
  *sys = buf.tms_stime / (double) ticks_per_sec;

  gettimeofday (&tp, 0);
  *wall = tp.tv_sec + 1.e-6*tp.tv_usec;

  return 0;
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

#ifdef DISABLE_TIMERS
  return 0;
#endif

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
  float utr_overhead;       /* overhead of calling underlying timing routine */
  bool found;               /* jump out of loop when name found */
  bool foundany;            /* whether summation print necessary */
  bool first;               /* flag 1st time entry found */

#ifdef DISABLE_TIMERS
  return 0;
#endif

  if ( ! initialized)
    return GPTLerror ("GPTLpr: GPTLinitialize() has not been called\n");

  if (get_thread_num (&nthreads, &maxthreads) > 0) 
    return GPTLerror ("GPTLpr: must only be called by master thread\n");

  if (id < 0 || id > 9999)
    return GPTLerror ("GPTLpr: bad id=%d for output file. Must be >= 0 and < 10000\n", id);

  sprintf (outfile, "timing.%d", id);

  if ( ! (fp = fopen (outfile, "w")))
    fp = stderr;

#ifdef NANOTIME
  fprintf (fp, "Clock rate = %f MHz\n", cpumhz);
#endif
  sum        = (float *) GPTLallocate (nthreads * sizeof (float));
  
  /*
  ** Determine underlying timing routine overhead to further refine per-timer 
  ** overhead estimate 
  */

  utr_overhead = utr_getoverhead ();

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
    GPTL_PAPIprstr (fp);
#endif

    fprintf (fp, "\n");        /* Done with titles, go to next line */

    for (ptr = timers[t]; ptr; ptr = ptr->next)
      printstats (ptr, fp, t, true, utr_overhead);

    /* Sum of overhead across timers is meaningful */

    sum[t]     = 0;
    totcount   = 0;
    totrecurse = 0;
    for (ptr = timers[t]; ptr; ptr = ptr->next) {
      sum[t]     += utr_norm (ptr->wall.overhead + ptr->count * 2 * utr_overhead);
      totcount   += ptr->count;
      totrecurse += ptr->nrecurse;
    }
    if (wallstats.enabled && overheadstats.enabled)
      fprintf (fp, "Overhead sum          = %9.3f wallclock seconds\n", sum[t]);
    fprintf (fp, "Total calls           = %u\n", totcount);
    fprintf (fp, "Total recursive calls = %u\n", totrecurse);
    if (totrecurse > 0)
      fprintf (fp, "Note: overhead computed only for non-recursive calls\n");
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
    GPTL_PAPIprstr (fp);
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
			float utr_overhead)     /* underlying timing routine overhead */
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
  float utrportion;    /* timer overhead due to gettimeofday only */
  float walloverhead;  /* wallclock overhead (sec) */

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
    elapse = utr_norm (utr_tofloat (&timer->wall.accum));
    wallmax = utr_norm (timer->wall.max);
    wallmin = utr_norm (timer->wall.min);
    fprintf (fp, "%9.3f %9.3f %9.3f ", elapse, wallmax, wallmin);

    /*
    ** Add cost of underlying timing routine to overhead est.  One factor of 2 
    ** is because both start and stop were called.  Other factor of 2 is because 
    ** underllying timing routine was called twice in each of start and stop.
    ** Adding a utrportion to walloverhead is due to estimate that a
    ** single utr_overhead is missing from each of start and stop.
    */

    if (overheadstats.enabled) {
      utrportion = utr_norm (timer->count * 2 * utr_overhead);
      walloverhead = utr_norm (timer->wall.overhead);
      fprintf (fp, "%9.3f %9.3f ", walloverhead + utrportion, 2*utrportion);
    }
  }

#ifdef HAVE_PAPI
  GPTL_PAPIpr (fp, &timer->aux, t, timer->count);
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
    tout->count       += tin->count;
    utr_sum (&tout->wall.accum, &tin->wall.accum);
    
    tout->wall.max = MAX (tout->wall.max, tin->wall.max);
    tout->wall.min = MIN (tout->wall.min, tin->wall.min);
    if (overheadstats.enabled)
      tout->wall.overhead += tin->wall.overhead;
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
  struct tms buf;

  (void) times (&buf);
  *usr = buf.tms_utime;
  *sys = buf.tms_stime;

  return 0;
}

#ifdef NUMERIC_TIMERS
/*
** getentry: find the entry in the hash table and return a pointer to it.
**
** Input args:
**   hashtable: the hashtable (array)
**   tag:       value to be hashed on (keep only 3 hex digits)
** Output args:
**   indx:      hashtable index
**
** Return value: pointer to the entry, or NULL if not found
*/

static inline Timer *getentry (const Hashentry *hashtable, /* hash table */
			       const unsigned long tag,    /* value to hash */
			       int *indx)                  /* hash index */
{
  int i;                 /* loop index */
  Timer *retval = 0;     /* value to be returned */

  /*
  ** Hash value is 3 hex digits.  Shift off the trailing 2 (or 3) bits
  ** because "tag" is likely to be an address, which is means a multiple
  ** of 4 on 32-bit addressable machines, and 8 on 64-bit.
  */

#ifdef BIT64
  *indx = (tag >> 3) & 0xFFF;
#else
  *indx = (tag >> 2) & 0xFFF;
#endif

  /* 
  ** If nument exceeds 1 there was a hash collision and we must search
  ** linearly through an array for a match
  */

  for (i = 0; i < hashtable[*indx].nument; i++) {
    if (tag == hashtable[*indx].entries[i]->tag) {
      retval = hashtable[*indx].entries[i];
      break;
    }
  }
  return retval;
}

#else

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
#endif

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
** line.  Currently only usable when GPTL compiled with NUMERIC_TIMERS
** defined.  Could do an sprintf of "this_fn" to a char var though when
** NUMERIC_TIMERS unset.  This way is much more efficient.
*/

#ifdef NUMERIC_TIMERS
#ifdef __cplusplus
extern "C" {
#endif

void __cyg_profile_func_enter (void *this_fn,
                               void *call_site)
{
  GPTLstart ((unsigned long) this_fn);
}

void __cyg_profile_func_exit (void *this_fn,
                              void *call_site)
{
  GPTLstop ((unsigned long) this_fn);
}

#ifdef __cplusplus
};
#endif
#endif

#ifdef NANOTIME
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

/* Call the underlying timing routine and return the info */

static inline void utr_get (UTRtype *tp)
{
  *tp = nanotime ();
}

/* Copy an underlying timing routine data value */

static inline void utr_save (UTRtype *tpout,
			     const UTRtype *tpin)
{
  *tpout = *tpin;
}

/* Difference two underlying timing routine data structures */

static inline void utr_get_delta (const UTRtype *tp1,
				  const UTRtype *tp2,
				  UTRtype *delta)
{
  *delta = *tp2 - *tp1;
}

/* Convert an underlying timing routine data structure to a float */

static inline float utr_tofloat (const UTRtype *val)
{
  return (float) *val;
}

/* Sum one underlying timing routine data structure into another */

static inline void utr_sum (UTRtype *tp1,
			    const UTRtype *tp2)
{
  *tp1 += *tp2;
}

/* Determine underlying timing routine overhead */

static float utr_getoverhead ()
{
  float nano_overhead;
  unsigned long long nanotime1;
  unsigned long long nanotime2;

  nanotime1 = nanotime ();
  nanotime2 = nanotime ();
  nano_overhead = nanotime2 - nanotime1;

  return nano_overhead;
}

/* Normalize cycles to seconds */

static float utr_norm (float val)
{
  return val / (cpumhz * 1.e6);
}

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

#else

static inline void utr_get (UTRtype *tp)
{
  (void) gettimeofday (tp, 0);
}

static inline void utr_save (UTRtype *tpout,
			     const UTRtype *tpin)
{
  tpout->tv_sec  = tpin->tv_sec;
  tpout->tv_usec = tpin->tv_usec;
}

static inline void utr_get_delta (const UTRtype *tp1,
				  const UTRtype *tp2,
				  UTRtype *delta)
{
  delta->tv_sec  = tp2->tv_sec - tp1->tv_sec;
  delta->tv_usec = tp2->tv_usec - tp1->tv_usec;
}

static inline float utr_tofloat (const UTRtype *val)
{
  return (float) val->tv_sec + 1.e-6*val->tv_usec;
}

static inline void utr_sum (UTRtype *tp1,
			    const UTRtype *tp2)
{
  tp1->tv_sec  += tp2->tv_sec;
  tp1->tv_usec += tp2->tv_usec;

    /*
    ** Adjust accumulated wallclock values to guard against overflow in the
    ** microsecond accumulator.
    */

  if (tp1->tv_usec > 10000000) {
    tp1->tv_sec  += 10;
    tp1->tv_usec -= 10000000;
  } else if (tp1->tv_usec < -10000000) {
    tp1->tv_sec  -= 10;
    tp1->tv_usec += 10000000;
  }
}

static float utr_getoverhead ()
{
  float overhead;
  struct timeval tp1;
  struct timeval tp2;

  gettimeofday (&tp1, 0);
  gettimeofday (&tp2, 0);
  overhead = (tp2.tv_sec  - tp1.tv_sec) + 
       1.e-6*(tp2.tv_usec - tp1.tv_usec);
  return overhead;
}

static float utr_norm (float val)
{
  return val;
}

#endif
