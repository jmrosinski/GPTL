#include <stdlib.h>  /* malloc */
#include <sys/time.h>      /* gettimeofday */
#include <sys/times.h>    /* times */
#include <unistd.h>        /* gettimeofday */
#include <stdio.h>
#include <string.h>        /* memset, strcmp (via STRMATCH) */
#if ( defined THREADED_OMP )
#include <omp.h>
#endif

#include "private.h"

static Timer **timers = 0;       /* linked list of timers */
static Timer **last = 0;         /* last element in list */
static int *max_depth;             /* maximum indentation level */
static int *max_name_len;        /* max length of timer name */
static int *current_depth;       /* current depth in timer tree */
static int nthreads            = 1;     /* num threads.  1 means no threading */
static bool initialized       = false; /* GPTinitialize has been called */
static Settings primary[] = {
  {GPTwall, "Wallclock","Wallclock max       min     Overhead  ", true },
  {GPTcpu,  "Cpu",      "Usr       sys       usr+sys",            false}};
static const int nprim = sizeof (primary) / sizeof (Settings);
static const int wallidx = 0;
static const int cpuidx = 1;
static bool wallenabled;
static bool cpuenabled;
static int naux = 0;               /* number of auxiliary stats */
static Settings aux[] = {{GPTother, "none", false}};

#if ( defined THREADED_OMP )
static omp_lock_t lock;
#endif

/* Local function prototypes */

static void printstats (Timer *, FILE *, int, bool);
static void *allocate (int);
static void add (Timer *, Timer *);
static int get_thread_num (void);
static int lock_mutex (void);
static int unlock_mutex (void);
static int get_cpustamp (long *, long *);

static long ticks_per_sec; /* clock ticks per second */

/*
** GPTsetoption: set option value to true or false.
**
** Input arguments:
**   option: option to be set
**   val:    value to which option should be set
**
** Return value: 0 (success) or -1 (failure)
*/

int GPTsetoption (Option option,     /* option name */
		  int val)           /* whether to enable */
{
  int n;   /* loop index */

  if (initialized)
    return (GPTerror ("GPTsetoption: Options must be set BEFORE GPTinitialize\n"));
  
  if (option == GPTabort_on_error) {
    GPTset_abort_on_error ((bool) val);
    printf ("GPTsetoption: setting abort on error flag to %d\n", val);
    return 0;
  }

  for (n = 0; n < nprim; n++) {
    if (primary[n].option == option) {
      primary[n].enabled = (bool) val;
      printf ("GPTsetoption: option %s set to %d\n", primary[n].name, (bool) val);
      return 0;
    }
  }

  for (n = 0; n < naux; n++) {
    if (aux[n].option == option) {
      aux[n].enabled = (bool) val;
      printf ("GPTsetoption: option %s set to %d\n", aux[n].name, val);
      return 0;
    }
  }
  return GPTerror ("GPTsetoption: option %d not found\n", (int) option);
}

/*
** GPTinitialize (): Initialization routine must be called from single-threaded
**   region before any other timing routines may be called.  The need for this
**   routine could be eliminated if not targetting timing library for threaded
**   capability. 
**
** return value: 0 (success) or -1 (failure)
*/

int GPTinitialize (void)
{
  int n;             /* index */

  if (initialized)
    return GPTerror ("GPTinitialize() has already been called\n");

#if ( defined THREADED_OMP )

  /*
  ** OMP: must call init_lock before using the lock (get_thread_num())
  */

  omp_init_lock (&lock);

  nthreads = omp_get_max_threads ();

#endif

  /*
  ** Allocate space for global arrays
  */

  timers        = (Timer **) allocate (nthreads * sizeof (Timer *));
  last          = (Timer **) allocate (nthreads * sizeof (Timer *));
  current_depth = (int *)    allocate (nthreads * sizeof (int));
  max_depth     = (int *)    allocate (nthreads * sizeof (int));
  max_name_len  = (int *)    allocate (nthreads * sizeof (int));

  /*
  ** Initialize array values
  */

  for (n = 0; n < nthreads; n++) {
    timers[n] = 0;
    current_depth[n] = 0;
    max_depth[n]     = 0;
    max_name_len[n]  = 0;
  }

  if (get_thread_num () > 0) 
    return GPTerror ("GPTinitialize: should only be called by master thread\n");

  /* Set enabled flags for speed */

  wallenabled = primary[wallidx].enabled;
  cpuenabled  = primary[cpuidx].enabled;

  if ((ticks_per_sec = sysconf (_SC_CLK_TCK)) == -1)
    return GPTerror ("GPTinitialize: sysconf (_SC_CLK_TCK) failed\n");

  initialized = true;
  return 0;
}

/*
** GPTstart: start a timer
**
** Input arguments:
**   name: timer name
**
** Return value: 0 (success) or -1 (failure)
*/

int GPTstart (char *name)       /* timer name */
{
  struct timeval tp1, tp2;      /* argument to gettimeofday */
  Timer *ptr;                   /* linked list pointer */

  int nchars;                   /* number of characters in timer */
  int mythread;                 /* thread index (of this thread) */
  int depth;                    /* depth in tree of timers which are on */

  /*
  ** 1st system timer call is solely for overhead timing
  */

  if (wallenabled)
    gettimeofday (&tp1, 0);

  if ( ! initialized)
    return GPTerror ("GPTstart: GPTinitialize has not been called\n");

  if ((mythread = get_thread_num ()) < 0)
    return GPTerror ("GPTstart\n");

  /*
  ** Look for the requested timer in the current list.  For those which don't
  ** match but are currently active, increase the indentation level by 1
  */

  depth = 0;
  for (ptr = timers[mythread]; ptr && ! STRMATCH (name, ptr->name); ptr = ptr->next) {
    if (ptr->onflg)
      ++depth;
  }

  if (ptr && ptr->onflg)
    return GPTerror ("GPTstart thread %d: timer %s was already on: "
		     "not restarting.\n", mythread, ptr->name);

  ++current_depth[mythread];
  if (current_depth[mythread] > max_depth[mythread])
    max_depth[mythread] = current_depth[mythread];

  /* 
  ** If a new thing is being timed, add a new link and initialize 
  */

  if ( ! ptr) {
    ptr = (Timer *) allocate (sizeof (Timer));
    memset (ptr, 0, sizeof (Timer));

    /* Truncate input name if longer than MAX_CHARS characters  */

    nchars = MIN (strlen (name), MAX_CHARS);
    max_name_len[mythread] = MAX (nchars, max_name_len[mythread]);

    ptr->name = (char *) allocate (nchars+1);
    strncpy (ptr->name, name, nchars);
    ptr->name[nchars] = '\0';
    ptr->depth = depth;

    if (timers[mythread])
      last[mythread]->next = ptr;
    else
      timers[mythread] = ptr;

    last[mythread] = ptr;

  } else {

    /*
    ** If computed indentation level is different than before or was
    ** already ambiguous, reset to ambiguous flag value.  This will likely
    ** happen any time the thing being timed is called from more than 1
    ** branch in the call tree.
    */

    if (ptr->depth != depth)
      ptr->depth = 0;
  }

  ptr->onflg = true;

  if (cpuenabled && get_cpustamp (&ptr->cpu.last_utime, &ptr->cpu.last_stime) < 0)
    return GPTerror ("GPTstart: get_cpustamp error");
  
  /*
  ** The 2nd system timer call is used both for overhead estimation and
  ** the input timer
  */
  
  if (wallenabled) {
    
    gettimeofday (&tp2, 0);
    ptr->wall.last_sec  = tp2.tv_sec;
    ptr->wall.last_usec = tp2.tv_usec;
    ptr->wall.overhead +=       (tp2.tv_sec  - tp1.tv_sec) + 
                          1.e-6*(tp2.tv_usec - tp1.tv_usec);
  }

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

int GPTstop (char *name)
{
  long delta_wtime_sec;     /* wallclock change fm GPTstart() to GPTstop() */    
  long delta_wtime_usec;    /* wallclock change fm GPTstart() to GPTstop() */
  float delta_wtime;        /* floating point wallclock change */
  struct timeval tp1, tp2;  /* argument to gettimeofday() */
  Timer *ptr;               /* linked list pointer */

  int mythread;             /* thread number for this process */

  long usr;
  long sys;

  /*
  ** The 1st system timer call is used both for overhead estimation and
  ** the input timer
  */

  if (wallenabled)
    gettimeofday (&tp1, 0);

  if (cpuenabled && get_cpustamp (&usr, &sys) < 0)
    return GPTerror (0);

  if ( ! initialized)
    return GPTerror ("GPTstop: GPTinitialize has not been called\n");

  if ((mythread = get_thread_num ()) < 0)
    return GPTerror ("GPTstop\n");

  for (ptr = timers[mythread]; ptr && ! STRMATCH (name, ptr->name); ptr = ptr->next);

  if ( ! ptr) 
    return GPTerror ("GPTstop: timer for %s had not been started.\n", name);

  if ( ! ptr->onflg )
    return GPTerror ("GPTstop: timer %s was already off.\n",ptr->name);

  --current_depth[mythread];

  ptr->onflg = false;
  ptr->count++;

  /*
  ** 1st timer stoppage: set max and min to computed values.  Otherwise apply
  ** max or min function
  */

  if (wallenabled) {

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

    if (ptr->wall.accum_usec > 1000000) {
      ptr->wall.accum_sec  += 1;
      ptr->wall.accum_usec -= 1000000;
    } else if (ptr->wall.accum_usec < -1000000) {
      ptr->wall.accum_sec  -= 1;
      ptr->wall.accum_usec += 1000000;
    }

    ptr->wall.last_sec  = tp1.tv_sec;
    ptr->wall.last_usec = tp1.tv_usec;

    /*
    ** 2nd system timer call is solely for overhead timing
    */

    gettimeofday (&tp2, 0);
    ptr->wall.overhead +=       (tp2.tv_sec  - tp1.tv_sec) + 
                          1.e-6*(tp2.tv_usec - tp1.tv_usec);
  }

  if (cpuenabled) {
    ptr->cpu.accum_utime += usr - ptr->cpu.last_utime;
    ptr->cpu.accum_stime += sys - ptr->cpu.last_stime;
    ptr->cpu.last_utime   = usr;
    ptr->cpu.last_stime   = sys;
  }

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
** Return value: 0 (success) or -1 (failure)
*/

int GPTstamp (double *wall, double *usr, double *sys)
{
  struct timeval tp;         /* argument to gettimeofday */
  struct tms buf;            /* argument to times */

  *usr = 0;
  *sys = 0;

  if (times (&buf) == -1)
    return GPTerror ("GPTstamp: times() failed. Timing bogus\n");

  *usr = buf.tms_utime / (double) ticks_per_sec;
  *sys = buf.tms_stime / (double) ticks_per_sec;

  gettimeofday (&tp, 0);
  *wall = tp.tv_sec + 1.e-6*tp.tv_usec;

  return 0;
}

/*
** GPTreset: reset all known timers to 0
**
** Return value: 0 (success) or -1 (failure)
*/

int GPTreset (void)
{
  int n;             /* index over threads */
  Timer *ptr;  /* linked list index */

  if ( ! initialized)
    return GPTerror ("GPTreset: GPTinitialize has not been called\n");

  /*
  ** Only allow the master thread to reset timers
  */

  if (get_thread_num () != 0)
    return 0;

  for (n = 0; n < nthreads; n++) {
    for (ptr = timers[n]; ptr; ptr = ptr->next) {
      memset (timers[n], 0, sizeof (Timer));
      printf ("Reset accumulators for timer %s to zero\n", ptr->name);
    }
  }
  return 0;
}

/* GPTpr: Print values of all timers */

int GPTpr (int id)
{
  FILE *fp;                /* file handle to write to */
  Timer *ptr;              /* walk through master thread linked list */
  Timer *tptr;             /* walk through slave threads linked lists */
  Timer sumstats;
  int n, nn;               /* indices */
  char outfile[11];        /* name of output file: timing.xxx */
  float sum;               /* sum of overhead values (per thread) */
  bool found;              /* jump out of loop when name found */
  bool foundany;           /* whether summation print necessary */
  bool first;              /* flag 1st time entry found */

  if ( ! initialized)
    return GPTerror ("GPTpr: GPTinitialize() has not been called\n");

  if (id < 0 || id > 999)
    return GPTerror ("GPTpr: id for output file must be >= 0 and < 1000\n");

  sprintf (outfile, "timing.%d", id);

  if ( ! (fp = fopen (outfile, "w")))
    fp = stderr;

  for (n = 0; n < nthreads; ++n) {
    fprintf (fp, "Stats for thread %d:\n", n);

    for (nn = 0; nn < max_depth[n]; ++nn)    /* max indent level (depth starts at 1) */
      fprintf (fp, "  ");

    for (nn = 0; nn < max_name_len[n]; ++nn) /* longest timer name */
      fprintf (fp, " ");

    fprintf (fp, "Called   ");

    for (nn = 0; nn < nprim; ++nn)
      if (primary[nn].enabled)
	fprintf (fp, "%s", primary[nn].str);

    for (nn = 0; nn < naux; ++nn)
      if (aux[nn].enabled)
	fprintf (fp, "%s", aux[nn].str);

    /* Done with titles, go to next line */

    fprintf (fp, "\n");

    for (ptr = timers[n]; ptr; ptr = ptr->next)
      printstats (ptr, fp, n, true);

    /* Sum of overhead across timers is meaningful */

    sum = 0;
    for (ptr = timers[n]; ptr; ptr = ptr->next)
      sum += ptr->wall.overhead;
    fprintf (fp, "Overhead sum = %9.3f wallclock seconds\n\n", sum);
  }

  /* Print per-name stats for all threads */

  if (nthreads > 1) {
    fprintf (fp, "\nSame stats sorted by timer for threaded regions:\n");
    fprintf (fp, "Thd ");

    for (nn = 0; nn < max_name_len[0]; ++nn) /* longest timer name */
      fprintf (fp, " ");

    fprintf (fp, "Called   ");

    for (nn = 0; nn < nprim; ++nn)
      if (primary[nn].enabled)
	fprintf (fp, "%s", primary[nn].str);

    for (nn = 0; nn < naux; ++nn)
      if (aux[nn].enabled)
	fprintf (fp, "%s", aux[nn].str);

    fprintf (fp, "\n");

    for (ptr = timers[0]; ptr; ptr = ptr->next) {
      
      /* To print sum stats, create a new timer, accumulate the */
      /* stats using the public "add" method, then invoke the print method.   */
      /* delete when done */

      foundany = false;
      first = true;
      for (n = 1; n < nthreads; ++n) {
	found = false;
	sumstats = *ptr;
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
  }
  return 0;
}

/* printstats: print a single timer */

static void printstats (Timer *timer,
			FILE *fp,
			int n,            /* thread number */
			bool doindent)         /* output stream */
{
  int i;
  int indent;
  int extraspace;
  long ticks_per_sec;
  float usr;
  float sys;
  float usrsys;
  float elapse;

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

  fprintf (fp, "%8ld", timer->count);

  if (wallenabled) {
    elapse = timer->wall.accum_sec + 1.e-6*timer->wall.accum_usec;
    fprintf (fp, "%9.3f %9.3f %9.3f ", elapse, timer->wall.max, timer->wall.min);
    fprintf (fp, "%9.3f", timer->wall.overhead);
  }

  if (cpuenabled) {
    usr = timer->cpu.accum_utime / (float) ticks_per_sec;
    sys = timer->cpu.accum_stime / (float) ticks_per_sec;
    usrsys = usr + sys;
    fprintf (fp, "%9.3f %9.3f %9.3f ", usr, sys, usrsys);
  }

  fprintf (fp, "\n");
}

static void *allocate (int nbytes)
{
  void *ptr = 0;

  if ( ! (ptr = malloc (nbytes)))
    (void) GPTerror ("allocate: malloc failed for %d bytes\n", nbytes);

  return ptr;
}

static void add (Timer *tout,   
		 Timer *tin)
{
  if (wallenabled) {
    tout->count           += tin->count;
    tout->wall.accum_sec  += tin->wall.accum_sec;
    tout->wall.accum_usec += tin->wall.accum_usec;
    if (tout->wall.accum_usec > 1000000) {
      tout->wall.accum_sec  += 1;
      tout->wall.accum_usec -= 1000000;
    }
    tout->wall.max       = MAX (tout->wall.max, tin->wall.max);
    tout->wall.min       = MIN (tout->wall.min, tin->wall.min);
    tout->wall.overhead += tin->wall.overhead;
  }

  if (cpuenabled) {
    tout->cpu.accum_utime += tin->cpu.accum_utime;
    tout->cpu.accum_stime += tin->cpu.accum_stime;
  }
}

/*
** get_thread_num: Obtain logical thread number of calling thread.  If new
** thread, adjust global variables.
*/

static int get_thread_num ()
{
  int mythread = 0;

#if ( defined THREADED_OMP )

  if ((mythread = omp_get_thread_num ()) >= nthreads)
    return GPTerror ("get_thread_num: returned id %d exceed nthreads %d\n",
		     mythread, nthreads);
#endif

  return mythread;
}

/*
** lock_mutex: lock a mutex for entry into a critical region
*/

static int lock_mutex (void)
{
#if ( defined THREADED_OMP )
  omp_set_lock (&lock);
#endif
  return 0;
}

/*
** unlock_mutex: unlock a mutex for exit from a critical region
*/

static int unlock_mutex (void)
{
#if ( defined THREADED_OMP )
  omp_unset_lock (&lock);
#endif
  return 0;
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

