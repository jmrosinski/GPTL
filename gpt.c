#include <stdlib.h>        /* malloc */
#include <sys/time.h>      /* gettimeofday */
#include <sys/times.h>     /* times */
#include <unistd.h>        /* gettimeofday */
#include <stdio.h>
#include <string.h>        /* memset, strcmp (via STRMATCH) */
#include <assert.h>

#include "private.h"

static Timer **timers = 0;       /* linked list of timers */
static Timer **last = 0;         /* last element in list */
static int *max_depth;           /* maximum indentation level */
static int *max_name_len;        /* max length of timer name */

typedef struct {
  unsigned int depth;            /* depth in calling tree */
  int padding[31];               /* padding is to mitigate false cache sharing */
} Nofalse; 
static Nofalse *current_depth;

static int nthreads    = -1;     /* num threads. Init to bad value */
static int maxthreads  = -1;     /* max threads (=nthreads for OMP). Init to bad value */
static bool initialized = false; /* GPTinitialize has been called */

typedef struct {
  const Option option;           /* wall, cpu, etc. */
  const char *str;               /* descriptive string for printing */
  bool enabled;                  /* flag */
} Settings;

static Settings cpustats =      {GPTcpu,      "Usr       sys       usr+sys   ", false};
static Settings wallstats =     {GPTwall,     "Wallclock max       min       ", true };
static Settings overheadstats = {GPToverhead, "Overhead  ",                     true };

static const int tablesize = 128*MAX_CHARS;  /* 128 is size of ASCII char set */
static Hashentry **hashtable;    /* table of entries hashed by sum of chars */
static unsigned int *novfl;      /* microsecond overflow counter (only when DIAG set */
static long ticks_per_sec;       /* clock ticks per second */

/* Local function prototypes */

static void printstats (const Timer *, FILE *, const int, const bool);
static void add (Timer *, const Timer *);
static int get_cpustamp (long *, long *);
static inline Timer *getentry (const Hashentry *, const char *, int *);

/*
** GPTsetoption: set option value to true or false.
**
** Input arguments:
**   option: option to be set
**   val:    value to which option should be set (nonzero=true, zero=false)
**
** Return value: 0 (success) or GPTerror (failure)
*/

int GPTsetoption (const int option,  /* option */
		  const int val)     /* whether to enable */
{
  int n;   /* loop index */

#ifdef DISABLE_TIMERS
  return 0;
#endif

  if (initialized)
    return GPTerror ("GPTsetoption: must be called BEFORE GPTinitialize\n");

  if (option == GPTabort_on_error) {
    GPTset_abort_on_error ((bool) val);
    printf ("GPTsetoption: setting abort on error flag to %d\n", val);
    return 0;
  }

  switch (option) {
  case GPTcpu:      
    cpustats.enabled = val; 
    printf ("GPTsetoption: set cpustats to %d\n", val);
    return 0;
  case GPTwall:     
    wallstats.enabled = val; 
    printf ("GPTsetoption: set wallstats to %d\n", val);
    return 0;
  case GPToverhead: 
    overheadstats.enabled = val; 
    printf ("GPTsetoption: set overheadstats to %d\n", val);
    return 0;
  default:
    break;
  }

#ifdef HAVE_PAPI
  if (GPT_PAPIsetoption (option, val) == 0)
    return 0;
#endif
  return GPTerror ("GPTsetoption: option %d not found\n", option);
}

/*
** GPTinitialize (): Initialization routine must be called from single-threaded
**   region before any other timing routines may be called.  The need for this
**   routine could be eliminated if not targetting timing library for threaded
**   capability. 
**
** return value: 0 (success) or GPTerror (failure)
*/

int GPTinitialize (void)
{
  int i, n;          /* indices */
  int ret;           /* return code */

#ifdef DISABLE_TIMERS
  return 0;
#endif

  if (initialized)
    return GPTerror ("GPTinitialize: has already been called\n");

  if (threadinit (&nthreads, &maxthreads) < 0)
    return GPTerror ("GPTinitialize: bad return from threadinit\n");

  if (get_thread_num (&nthreads, &maxthreads) > 0) 
    return GPTerror ("GPTinitialize: must only be called by master thread\n");

  if ((ticks_per_sec = sysconf (_SC_CLK_TCK)) == -1)
    return GPTerror ("GPTinitialize: sysconf (_SC_CLK_TCK) failed\n");

  /* Allocate space for global arrays */

  timers        = (Timer **)     GPTallocate (maxthreads * sizeof (Timer *));
  last          = (Timer **)     GPTallocate (maxthreads * sizeof (Timer *));
  current_depth = (Nofalse *)    GPTallocate (maxthreads * sizeof (Nofalse));
  max_depth     = (int *)        GPTallocate (maxthreads * sizeof (int));
  max_name_len  = (int *)        GPTallocate (maxthreads * sizeof (int));
  hashtable     = (Hashentry **) GPTallocate (maxthreads * sizeof (Hashentry *));
#ifdef DIAG
  novfl         = (unsigned int *) GPTallocate (maxthreads * sizeof (unsigned int));
#endif

  /* Initialize array values */

  for (n = 0; n < maxthreads; n++) {
    timers[n] = 0;
    current_depth[n].depth = 0;
    max_depth[n]     = 0;
    max_name_len[n]  = 0;
    hashtable[n] = (Hashentry *) GPTallocate (tablesize * sizeof (Hashentry));
#ifdef DIAG
    novfl[n] = 0;
#endif
    for (i = 0; i < tablesize; i++) {
      hashtable[n][i].nument = 0;
      hashtable[n][i].entries = 0;
    }
  }

#ifdef HAVE_PAPI
  if (GPT_PAPIinitialize (maxthreads) < 0)
    return GPTerror ("GPTinitialize: GPT_PAPIinitialize failure\n");
#endif

  initialized = true;
  return 0;
}

/*
** GPTfinalize (): Finalization routine must be called from single-threaded
**   region. Free all malloc'd space
**
** return value: 0 (success) or GPTerror (failure)
*/

int GPTfinalize (void)
{
  int n;                /* index */
  Timer *ptr, *ptrnext; /* ll indices */

#ifdef DISABLE_TIMERS
  return 0;
#endif

  if ( ! initialized)
    return GPTerror ("GPTfinalize: GPTinitialize() has not been called\n");

  if (get_thread_num (&nthreads, &maxthreads) > 0) 
    return GPTerror ("GPTfinalize: must only be called by master thread\n");

  for (n = 0; n < maxthreads; ++n) {
    free (hashtable[n]);
    for (ptr = timers[n]; ptr; ptr = ptrnext) {
      ptrnext = ptr->next;
      free (ptr);
    }
  }

  free (timers);
  free (current_depth);
  free (max_depth);
  free (max_name_len);
  free (hashtable);
#ifdef DIAG
  free (novfl);
#endif

  threadfinalize ();
#ifdef HAVE_PAPI
  GPT_PAPIfinalize (maxthreads);
#endif
  initialized = false;
  return 0;
}

/*
** GPTstart: start a timer
**
** Input arguments:
**   name: timer name
**
** Return value: 0 (success) or GPTerror (failure)
*/

int GPTstart (const char *name)       /* timer name */
{
  struct timeval tp1, tp2;      /* argument returned from gettimeofday */
  Timer *ptr;                   /* linked list pointer */
  Timer **eptr;                 /* for realloc */

  int nchars;                   /* number of characters in timer */
  int mythread;                 /* thread index (of this thread) */
  int indx;                     /* hash table index */
  int nument;                   /* number of entries for a hash collision */

#ifdef DISABLE_TIMERS
  return 0;
#endif

  if ((mythread = get_thread_num (&nthreads, &maxthreads)) < 0)
    return GPTerror ("GPTstart\n");

  /* 1st calls to overheadstart and gettimeofday are solely for overhead timing */

  if (overheadstats.enabled) {
#ifdef HAVE_PAPI
    (void) GPT_PAPIoverheadstart (mythread);
#endif

    if (wallstats.enabled)
      gettimeofday (&tp1, 0);
  }

  if ( ! initialized)
    return GPTerror ("GPTstart: GPTinitialize has not been called\n");

  /* Look for the requested timer in the current list. */

#ifdef HASH
  ptr = getentry (hashtable[mythread], name, &indx);
  assert (indx < tablesize);
#else
  for (ptr = timers[mythread]; ptr && ! STRMATCH (name, ptr->name); ptr = ptr->next);
#endif

  if (ptr && ptr->onflg)
    return GPTerror ("GPTstart thread %d: timer %s was already on: "
		     "not restarting.\n", mythread, ptr->name);

  ++current_depth[mythread].depth;
  if (current_depth[mythread].depth > max_depth[mythread])
    max_depth[mythread] = current_depth[mythread].depth;

  /* If a new thing is being timed, add a new entry and initialize */

  if ( ! ptr) {
    ptr = (Timer *) GPTallocate (sizeof (Timer));
    memset (ptr, 0, sizeof (Timer));

    /* Truncate input name if longer than MAX_CHARS characters  */

    nchars = MIN (strlen (name), MAX_CHARS);
    max_name_len[mythread] = MAX (nchars, max_name_len[mythread]);

    strncpy (ptr->name, name, nchars);
    ptr->name[nchars] = '\0';
    ptr->depth = current_depth[mythread].depth;

    if (timers[mythread])
      last[mythread]->next = ptr;
    else
      timers[mythread] = ptr;

    last[mythread] = ptr;

#ifdef HASH
    ++hashtable[mythread][indx].nument;
    nument = hashtable[mythread][indx].nument;

    eptr = realloc (hashtable[mythread][indx].entries, nument * sizeof (Timer *));
    if ( ! eptr)
      return GPTerror ("GPTstart: realloc error\n");

    hashtable[mythread][indx].entries = eptr;						 
    hashtable[mythread][indx].entries[nument-1] = ptr;
#endif

  } else {

    /*
    ** If computed indentation level is different than before or was
    ** already ambiguous, reset to ambiguous flag value.  This will likely
    ** happen any time the thing being timed is called from more than 1
    ** branch in the call tree.
    */
  
    if (ptr->depth != current_depth[mythread].depth)
      ptr->depth = 0;
  }

  ptr->onflg = true;

  if (cpustats.enabled && get_cpustamp (&ptr->cpu.last_utime, &ptr->cpu.last_stime) < 0)
    return GPTerror ("GPTstart: get_cpustamp error");
  
  /*
  ** The 2nd system timer call is used both for overhead estimation and
  ** the input timer
  */
  
  if (wallstats.enabled) {
    gettimeofday (&tp2, 0);
    ptr->wall.last_sec  = tp2.tv_sec;
    ptr->wall.last_usec = tp2.tv_usec;
    if (overheadstats.enabled)
      ptr->wall.overhead +=       (tp2.tv_sec  - tp1.tv_sec) + 
                            1.e-6*(tp2.tv_usec - tp1.tv_usec);
  }

#ifdef HAVE_PAPI
  if (GPT_PAPIstart (mythread, &ptr->aux) < 0)
    return GPTerror ("GPTstart: error from GPT_PAPIstart\n");

  if (overheadstats.enabled)
    (void) GPT_PAPIoverheadstop (mythread, &ptr->aux);
#endif

  return (0);
}

/*
** GPTstop: stop a timer
**
** Input arguments:
**   name: timer name
**
** Return value: 0 (success) or -1 (failure)
*/

int GPTstop (const char *name) /* timer name */
{
  long delta_wtime_sec;     /* wallclock sec change fm GPTstart() to GPTstop() */    
  long delta_wtime_usec;    /* wallclock usec change fm GPTstart() to GPTstop() */
  float delta_wtime;        /* floating point wallclock change */
  struct timeval tp1, tp2;  /* argument to gettimeofday() */
  Timer *ptr;               /* linked list pointer */

  int mythread;             /* thread number for this process */
  int indx;                 /* index into hash table */

  long usr;                 /* user time (returned from get_cpustamp) */
  long sys;                 /* system time (returned from get_cpustamp) */

#ifdef DISABLE_TIMERS
  return 0;
#endif

  if ((mythread = get_thread_num (&nthreads, &maxthreads)) < 0)
    return GPTerror ("GPTstop\n");

#ifdef HAVE_PAPI
  if (overheadstats.enabled)
    (void) GPT_PAPIoverheadstart (mythread);
#endif

  /*
  ** The 1st system timer call is used both for overhead estimation and
  ** the input timer
  */
    
  if (wallstats.enabled)
    gettimeofday (&tp1, 0);

  if (cpustats.enabled && get_cpustamp (&usr, &sys) < 0)
    return GPTerror (0);

  if ( ! initialized)
    return GPTerror ("GPTstop: GPTinitialize has not been called\n");

#ifdef HASH
  ptr = getentry (hashtable[mythread], name, &indx);
#else
  for (ptr = timers[mythread]; ptr && ! STRMATCH (name, ptr->name); ptr = ptr->next);
#endif

  if ( ! ptr) 
    return GPTerror ("GPTstop: timer for %s had not been started.\n", name);

  if ( ! ptr->onflg )
    return GPTerror ("GPTstop: timer %s was already off.\n",ptr->name);

#ifdef HAVE_PAPI
  if (GPT_PAPIstop (mythread, &ptr->aux) < 0)
    return GPTerror ("GPTstart: error from GPT_PAPIstop\n");
#endif

  --current_depth[mythread].depth;

  ptr->onflg = false;
  ptr->count++;

  if (wallstats.enabled) {

    delta_wtime_sec  = tp1.tv_sec  - ptr->wall.last_sec;
    delta_wtime_usec = tp1.tv_usec - ptr->wall.last_usec;
    delta_wtime      = delta_wtime_sec + 1.e-6*delta_wtime_usec;

    if (ptr->count == 1) {
      ptr->wall.max = delta_wtime;
      ptr->wall.min = delta_wtime;
    } else {
      ptr->wall.max = MAX (ptr->wall.max, delta_wtime);
      ptr->wall.min = MIN (ptr->wall.min, delta_wtime);
    }

    ptr->wall.accum_sec  += delta_wtime_sec;
    ptr->wall.accum_usec += delta_wtime_usec;

    /*
    ** Adjust accumulated wallclock values to guard against overflow in the
    ** microsecond accumulator.
    */

    if (ptr->wall.accum_usec > 10000000) {
      ptr->wall.accum_sec  += 10;
      ptr->wall.accum_usec -= 10000000;
#ifdef DIAG
      ++novfl[mythread];
#endif
    } else if (ptr->wall.accum_usec < -10000000) {
      ptr->wall.accum_sec  -= 10;
      ptr->wall.accum_usec += 10000000;
#ifdef DIAG
      ++novfl[mythread];
#endif
    }

    /* 2nd system timer call is solely for overhead timing */

    if (overheadstats.enabled) {
      gettimeofday (&tp2, 0);
      ptr->wall.overhead +=       (tp2.tv_sec  - tp1.tv_sec) + 
                            1.e-6*(tp2.tv_usec - tp1.tv_usec);
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
    (void) GPT_PAPIoverheadstop (mythread, &ptr->aux);
#endif

  return 0;
}

/*
** GPTstamp: Compute timestamp of usr, sys, and wallclock time (seconds)
**
** Output arguments:
**   wall: wallclock
**   usr:  user time
**   sys:  system time
**
** Return value: 0 (success) or GPTerror (failure)
*/

int GPTstamp (double *wall, double *usr, double *sys)
{
  struct timeval tp;         /* argument to gettimeofday */
  struct tms buf;            /* argument to times */

#ifdef DISABLE_TIMERS
  return 0;
#endif

  *usr = 0;
  *sys = 0;

  if (times (&buf) == -1)
    return GPTerror ("GPTstamp: times() failed. Results bogus\n");

  *usr = buf.tms_utime / (double) ticks_per_sec;
  *sys = buf.tms_stime / (double) ticks_per_sec;

  gettimeofday (&tp, 0);
  *wall = tp.tv_sec + 1.e-6*tp.tv_usec;

  return 0;
}

/*
** GPTreset: reset all known timers to 0
**
** Return value: 0 (success) or GPTerror (failure)
*/

int GPTreset (void)
{
  int n;             /* index over threads */
  Timer *ptr;        /* linked list index */

#ifdef DISABLE_TIMERS
  return 0;
#endif

  if ( ! initialized)
    return GPTerror ("GPTreset: GPTinitialize has not been called\n");

  /* Only allow the master thread to reset timers */

  if (get_thread_num (&nthreads, &maxthreads) > 0) 
    return GPTerror ("GPTreset: must only be called by master thread\n");

  for (n = 0; n < nthreads; n++) {
    for (ptr = timers[n]; ptr; ptr = ptr->next) {
      ptr->onflg = false;
      ptr->count = 0;
      memset (&ptr->wall, 0, sizeof (ptr->wall));
      memset (&ptr->cpu, 0, sizeof (ptr->cpu));
#ifdef HAVE_PAPI
      memset (&ptr->aux, 0, sizeof (ptr->aux));
#endif
    }
  }
  printf ("GPTreset: accumulators for all timers set to zero\n");
  return 0;
}

/* 
** GPTpr: Print values of all timers
**
** Input arguments:
**   id: integer to append to string "timing."
**
** Return value: 0 (success) or GPTerror (failure)
*/

int GPTpr (const int id)   /* output file will be named "timing.<id>" */
{
  FILE *fp;                /* file handle to write to */
  Timer *ptr;              /* walk through master thread linked list */
  Timer *tptr;             /* walk through slave threads linked lists */
  Timer sumstats;          /* sum of same timer stats over threads */
  int i, ii, n, nn;        /* indices */
  char outfile[11];        /* name of output file: timing.xxx */
  float *sum;              /* sum of overhead values (per thread) */
  float osum;              /* sum of overhead over threads */
  bool found;              /* jump out of loop when name found */
  bool foundany;           /* whether summation print necessary */
  bool first;              /* flag 1st time entry found */

#ifdef DISABLE_TIMERS
  return 0;
#endif

  if ( ! initialized)
    return GPTerror ("GPTpr: GPTinitialize() has not been called\n");

  if (id < 0 || id > 999)
    return GPTerror ("GPTpr: id for output file must be >= 0 and < 1000\n");

  sprintf (outfile, "timing.%d", id);

  if ( ! (fp = fopen (outfile, "w")))
    fp = stderr;

  sum = GPTallocate (nthreads * sizeof (float));

  for (n = 0; n < nthreads; ++n) {
    if (n > 0)
      fprintf (fp, "\n");
    fprintf (fp, "Stats for thread %d:\n", n);

    for (nn = 0; nn < max_depth[n]; ++nn)    /* max indent level (depth starts at 1) */
      fprintf (fp, "  ");
    for (nn = 0; nn < max_name_len[n]; ++nn) /* longest timer name */
      fprintf (fp, " ");

    fprintf (fp, "Called   ");

    /* Print strings for enabled timer types */

    if (cpustats.enabled)
      fprintf (fp, "%s", cpustats.str);
    if (wallstats.enabled) {
      fprintf (fp, "%s", wallstats.str);
      if (overheadstats.enabled)
	fprintf (fp, "%s", overheadstats.str);
    }

#ifdef HAVE_PAPI
    GPT_PAPIprstr (fp);
#endif

    /* Done with titles, go to next line */

    fprintf (fp, "\n");

    for (ptr = timers[n]; ptr; ptr = ptr->next)
      printstats (ptr, fp, n, true);

    /* Sum of overhead across timers is meaningful */

    if (wallstats.enabled && overheadstats.enabled) {
      sum[n] = 0;
      for (ptr = timers[n]; ptr; ptr = ptr->next)
	sum[n] += ptr->wall.overhead;
      fprintf (fp, "Overhead sum = %9.3f wallclock seconds\n", sum[n]);
    }
  }

  /* Print per-name stats for all threads */

  if (nthreads > 1) {
    fprintf (fp, "\nSame stats sorted by timer for threaded regions:\n");
    fprintf (fp, "Thd ");

    for (nn = 0; nn < max_name_len[0]; ++nn) /* longest timer name */
      fprintf (fp, " ");

    fprintf (fp, "Called   ");

    if (cpustats.enabled)
      fprintf (fp, "%s", cpustats.str);
    if (wallstats.enabled) {
      fprintf (fp, "%s", wallstats.str);
      if (overheadstats.enabled)
	fprintf (fp, "%s", overheadstats.str);
    }

#ifdef HAVE_PAPI
    GPT_PAPIprstr (fp);
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
      for (n = 1; n < nthreads; ++n) {
	found = false;
	for (tptr = timers[n]; tptr && ! found; tptr = tptr->next) {
	  if (STRMATCH (ptr->name, tptr->name)) {

	    /* Only print thread 0 when this timer found for other threads */

	    if (first) {
	      first = false;
	      fprintf (fp, "%3.3d ", 0);
	      printstats (ptr, fp, 0, false);
	    }

	    found = true;
	    foundany = true;
	    fprintf (fp, "%3.3d ", n);
	    printstats (tptr, fp, 0, false);
	    add (&sumstats, tptr);
	  }
	}
      }

      if (foundany) {
	fprintf (fp, "SUM ");
	printstats (&sumstats, fp, 0, false);
	fprintf (fp, "\n");
      }
    }

    /* Repeat overhead print in loop over threads */

    if (wallstats.enabled && overheadstats.enabled) {
      osum = 0.;
      for (n = 0; n < nthreads; ++n) {
	fprintf (fp, "OVERHEAD.%3.3d (wallclock seconds) = %9.3f\n", n, sum[n]);
	osum += sum[n];
      }
      fprintf (fp, "OVERHEAD.SUM (wallclock seconds) = %9.3f\n", osum);
    }
  }

#ifdef DIAG
  fprintf (fp, "\n");
  for (n = 0; n < nthreads; ++n) 
    fprintf (fp, "novfl[%d]=%d\n", n, novfl[n]);
#endif

  /* Print hash table stats */

#ifdef HASH  
  for (n = 0; n < nthreads; n++) {
    first = true;
    for (i = 0; i < tablesize; i++) {
      int nument = hashtable[n][i].nument;
      if (nument > 1) {
	if (first) {
	  first = false;
	  fprintf (fp, "\nthread %d had some hash collisions:\n", n);
	}
	fprintf (fp, "hashtable[%d][%d] had %d entries:", n, i, nument);
	for (ii = 0; ii < nument; ii++)
	  fprintf (fp, " %s", hashtable[n][i].entries[ii]->name);
	fprintf (fp, "\n");
      }
    }
  }
#endif

  free (sum);
  return 0;
}

/* 
** printstats: print a single timer
**
** Input arguments:
**   timer:    timer for which to print stats
**   fp:       file descriptor to write to
**   n:        thread number
**   doindent: whether to indent
*/

static void printstats (const Timer *timer,     /* timer to print */
			FILE *fp,               /* file descriptor to write to */
			const int n,            /* thread number */
			const bool doindent)    /* whether indenting will be done */
{
  int i;               /* index */
  int indent;          /* index for indenting */
  int extraspace;      /* for padding to length of longest name */
  long ticks_per_sec;  /* returned from sysconf */
  float usr;           /* user time */
  float sys;           /* system time */
  float usrsys;        /* usr + sys */
  float elapse;        /* elapsed time */

  if ((ticks_per_sec = sysconf (_SC_CLK_TCK)) == -1)
    (void) GPTerror ("printstats: token _SC_CLK_TCK is not defined\n");

  /* Indent to depth of this timer */

  if (doindent)
    for (indent = 0; indent < timer->depth; ++indent)  /* depth starts at 1 */
      fprintf (fp, "  ");

  fprintf (fp, "%s", timer->name);

  /* Pad to length of longest name */

  extraspace = max_name_len[n] - strlen (timer->name);
  for (i = 0; i < extraspace; ++i)
    fprintf (fp, " ");

  /* Pad to max indent level */

  if (doindent)
    for (indent = timer->depth; indent < max_depth[n]; ++indent)
      fprintf (fp, "  ");

  fprintf (fp, "%8ld ", timer->count);

  if (cpustats.enabled) {
    usr = timer->cpu.accum_utime / (float) ticks_per_sec;
    sys = timer->cpu.accum_stime / (float) ticks_per_sec;
    usrsys = usr + sys;
    fprintf (fp, "%9.3f %9.3f %9.3f ", usr, sys, usrsys);
  }

  if (wallstats.enabled) {
    elapse = timer->wall.accum_sec + 1.e-6*timer->wall.accum_usec;
    fprintf (fp, "%9.3f %9.3f %9.3f ", elapse, timer->wall.max, timer->wall.min);
    if (overheadstats.enabled)
      fprintf (fp, "%9.3f ", timer->wall.overhead);
  }

#ifdef HAVE_PAPI
  GPT_PAPIpr (fp, &timer->aux);
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
    tout->count           += tin->count;
    tout->wall.accum_sec  += tin->wall.accum_sec;
    tout->wall.accum_usec += tin->wall.accum_usec;
    if (tout->wall.accum_usec > 10000000) {
      tout->wall.accum_sec  += 10;
      tout->wall.accum_usec -= 10000000;
    } else if (tout->wall.accum_usec < -10000000) {
      tout->wall.accum_sec  -= 10;
      tout->wall.accum_usec += 10000000;
    }
      
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
  GPT_PAPIadd (&tout->aux, &tin->aux);
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

static int get_cpustamp (long *usr, long *sys)
{
  struct tms buf;

  (void) times (&buf);
  *usr = buf.tms_utime;
  *sys = buf.tms_stime;

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

  /* Generate the hash value by summing values of the chars in "name */

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
