#include <stdlib.h>  /* malloc */
#include <sys/time.h>      /* gettimeofday */
#include <sys/times.h>    /* times */
#include <unistd.h>        /* gettimeofday */
#include <stdio.h>
#include <string.h>        /* memset, strcmp (via STRMATCH) */

#include "gpt.h"
#include "private.h"

static struct node **timers = NULL;       /* linked list of timers */
static struct node **last = NULL;         /* last element in list */
static float *overhead;                   /* wallclock estimate of timer overhead */
static int *max_indent_level;             /* maximum indentation level */
static int numthreads            = 1;     /* num threads.  1 means no threading */
static Boolean initialized       = false; /* GPTinitialize has been called */
static Boolean wallenabled       = false; /* wallclock timer stats enabled */
static Boolean usrsysenabled     = false; /* usr & sys timer stats enabled */
static Boolean pclenabled        = false; /* need to call PCL library */     
static Boolean pcl_cyclesenabled = false; /* enable PCL cycle count */
static int pcl_cyclesindex       = -1;    /* index for PCL cycle count */

typedef struct {
  OptionName name;
  Boolean enabled;
  char string[10];
  int index;
} Counter_t;

Counter_t counter[] = {
  {usrsys,               true,  "          ", -1},
  {wall,                 true,  "Wallclock ", -1},
#ifdef HAVE_PCL
  {pcl_start,            false, "          ", -1},  /* bracket PCL entries */
  {pcl_l1dcache_miss,    false, "l1 D miss ", -1},
  {pcl_l2cache_miss,     false, "L2 miss   ", -1},
  {pcl_cycles,           false, "Cycles    ", -1},
  {pcl_elapsed_cycles,   false, "E-Cycles  ", -1},
  {pcl_fp_instr,         false, "FP instr  ", -1},
  {pcl_loadstore_instr,  false, "L/S instr ", -1},
  {pcl_instr,            false, "Instruct  ", -1},
  {pcl_stall,            false, "Stall     ", -1},
  {pcl_end,              false, "          ", -1},  /* bracket PCL entries */
#endif
};

int ncounter = sizeof (counter) / sizeof (Counter_t);

/*
** Needed by PCL library: otherwise unused
*/

PCL_DESCR_TYPE *descr;
int pcl_counter_list[PCL_COUNTER_MAX];
int npcl = 0;                      /* number of PCL counters enabled */
PCL_CNT_TYPE *overhead_pcl;        /* overhead counter (cycles) */

/*
** GPTsetoption: set option value to true or false.
**
** Input arguments:
**   option: option to be set
**   val:    value to which option should be set
**
** Return value: 0 (success) or -1 (failure)
*/

/* 
** Specific to GPTpr
*/

static void fillstats (struct Stats *, struct node *);
static void print_header (FILE *, int);
static void print_stats_line (FILE *, struct Stats *);

static long ticks_per_sec; /* clock ticks per second */

/*******************************************************************************/

int GPTsetoption (OptionName option, Boolean val)
{
  int n;

  if (initialized)
    return (GPTerror ("GPTsetoption: Options must be set BEFORE GPTinitialize\n"));

  for (n = 0; n < ncounter; n++) {
    if (counter[n].name == option) {
      counter[n].enabled = val;

      if (val)
	printf ("GPTsetoption: option enabled:  %s\n", counter[n].string);
      else
	printf ("GPTsetoption: option disabled: %s\n", counter[n].string);

      return 0;
    }
  }

  return (GPTerror ("t_setoption: Option with enum index %d not available\n",
		     option));
}

/*
** GPTinitialize (): Initialization routine must be called from single-threaded
**   region before any other timing routines may be called.  The need for this
**   routine could be eliminated if not targetting timing library for threaded
**   capability. 
**
** return value: 0 (success) or -1 (failure)
*/

int t_initialize ()
{
#if ( defined THREADED_OMP )

omp_lock_t lock;

#elif ( defined THREADED_PTHREADS )

pthread_mutex_t t_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_t *threadid;

#endif

  int n;             /* index */
  int nbytes;        /* number of bytes for malloc */
  int ret;           /* return code */

  if (initialized)
    return GPTerror ("GPTinitialize has already been called\n");

#if ( defined THREADED_OMP )

  /*
  ** OMP: must call init_lock before using the lock (get_thread_num())
  */

  omp_init_lock (&lock);

  numthreads = omp_get_max_threads();

#elif ( defined THREADED_PTHREADS )

  numthreads = MAX_THREADS;

#endif

  /*
  ** Allocate space for global arrays
  */

  nbytes = numthreads * sizeof (struct node *);
  if ((timers = (struct node **) malloc (nbytes)) == 0)
    return GPTerror ("malloc failure: %d items\n", numthreads);

  if ((last = (struct node **) malloc (nbytes)) == 0)
    return GPTerror ("malloc failure: %d items\n", numthreads);

  nbytes = numthreads * sizeof (float);
  if ((overhead = (float *) malloc (nbytes)) == 0)
    return GPTerror ("malloc failure: %d items\n", numthreads);

  nbytes = numthreads * sizeof (PCL_CNT_TYPE);
  if ((overhead_pcl = (PCL_CNT_TYPE *) malloc (nbytes)) == 0)
    return GPTerror ("malloc failure: %d items\n", numthreads);

  nbytes = numthreads * sizeof (int);
  if ((max_indent_level = (int *) malloc (nbytes)) == 0)
    return GPTerror ("malloc failure for %d items\n", numthreads);

  /*
  ** Initialize array values
  */

  for (n = 0; n < numthreads; n++) {
    timers[n] = 0;
    last[n] = 0;
    overhead[n] = 0.;
    overhead_pcl[n] = 0;
    max_indent_level[n] = 0;
  }

#ifdef THREADED_PTHREADS

  /*
  ** In the pthreads case, we must manage the threadid array which maps
  ** physical thread id's to logical id's
  */

  nbytes = numthreads * sizeof (pthread_t);
  if ((threadid = (pthread_t *) malloc (nbytes)) == 0)
    return GPTerror ("malloc failure for %d items\n", numthreads);

  /*
  ** Reset numthreads to 1 and define the threadid array now that initialization 
  ** is done.
  */

  threadid[0] = pthread_self ();
  numthreads = 1;

#endif

  if (get_thread_num () > 0) 
    return GPTerror ("GPTinitialize: should only be called by master thread\n");

  for (n = 0; n < npossible; n++) {
    if (counter[n].enabled) {
      if (counter[n].name == usrsys)
	usrsysenabled = true;

      if (counter[n].name == wall)
	wallenabled = true;

#ifdef HAVE_PCL

      /*
      ** Set up PCL stuff based on what setoption has provided.
      */

      if (counter[n]->name > pcl_start && counter[n]->name < pcl_end) {

	pclenabled = true;

	switch (counter[n].name) {

	case pcl_l1dcache_miss:
	  counter_list[npcl++] = PCL_L1DCACHE_MISS;
	  break;
	  
	case pcl_l2cache_miss: 
	  counter_list[npcl++] = PCL_L2CACHE_MISS;
	  break;
	  
	case pcl_cycles: 
	  pcl_cyclesindex = npcl;
	  pcl_cyclesenabled = true;
	  counter_list[pcl++] = PCL_CYCLES;
	  break;

	case pcl_elapsed_cycles: 
	  counter_list[npcl++] = PCL_ELAPSED_CYCLES;
	  break;

	case pcl_fp_instr: 
	  counter_list[npcl++] = PCL_FP_INSTR;
	  break;

	case pcl_loadstore_instr: 
	  counter_list[npcl++] = PCL_LOADSTORE_INSTR;
	  break;

	case pcl_instr: 
	  counter_list[npcl++] = PCL_INSTR;
	  break;

	case pcl_stall: 
	  counter_list[npcl++] = PCL_STALL;
	  break;
	
	default:
	  break;

	}
      }
#endif
    }
  }

#ifdef HAVE_PCL

  if (ncpl > 0) {
    int thread;         /* thread number */

    nbytes = numthreads * sizeof (PCL_DESCR_TYPE);
    if ((descr = (PCL_DESCR_TYPE *) malloc (nbytes)) == 0)
      return GPTerror ("malloc failure: %d items\n", numthreads);

    /*
    ** PCLinit must be called on a per-thread basis.  Therefore must make the call here
    ** rather than in t_initialize.  null timer list flags not initialized.
    ** Also, the critical section is necessary because PCLstart appears not to be
    ** thread-safe.
    */

#pragma omp parallel for
    
    for (thread = 0; thread < numthreads; thread++) {

      unsigned int flags;           /* mode flags needed by PCL */

#pragma omp critical

      {
	if ((ret = PCLinit (&descr[thread])) != PCL_SUCCESS)
	  return GPTerror ("unable to allocate PCL handle for thread %d. %s\n",
			  thread, t_pclstr (ret));

	/*
	** Always count user mode only
	*/
      
	flags = PCL_MODE_USER;

	if ((ret = PCLquery (descr[thread], counter_list, npcl, flags)) != PCL_SUCCESS)
	  return GPTerror ("Bad return from PCLquery thread %d: %s\n", thread, t_pclstr (ret));

	if ((ret = PCLstart (descr[thread], counter_list, npcl, flags)) != PCL_SUCCESS)
	  return GPTerror ("PCLstart failed thread=%d: %s\n", thread, t_pclstr (ret));
      }
    }
  }
#endif
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

int GPTstart (char *name)        /* timer name */
{
  struct timeval tp1, tp2;      /* argument to gettimeofday */
  struct node *ptr;             /* linked list pointer */

  int numchars;                 /* number of characters in timer */
  int mythread;                 /* thread index (of this thread) */
  int indent_level = 0;         /* required indentation level for this timer */
  int ret;                      /* return code */

  PCL_CNT_TYPE i_pcl_result1[PCL_COUNTER_MAX];     /* init. output fm PCLread */
  PCL_CNT_TYPE i_pcl_result2[PCL_COUNTER_MAX];     /* final output fm PCLread */
  PCL_FP_CNT_TYPE fp_pcl_result[PCL_COUNTER_MAX];  /* required by PCLread */

  /*
  ** 1st system timer call is solely for overhead timing
  */

  if (wallenabled)
    gettimeofday (&tp1, NULL);

  if ( ! initialized)
    return GPTerror ("GPTstart: t_initialize has not been called\n");

  if ((mythread = get_thread_num ()) < 0)
    return GPTerror ("GPTstart\n");

  if (npcl > 0) {
    ret = PCLread (descr[mythread], i_pcl_result1, fp_pcl_result, npcl);
    if (ret != PCL_SUCCESS)
      return GPTerror ("GPTstart: error from PCLread: %s\n", t_pclstr (ret));
  }

  /*
  ** Look for the requested timer in the current list.  For those which don't
  ** match but are currently active, increase the indentation level by 1
  */

  for (ptr = timers[mythread]; ptr != NULL && ! STRMATCH (name, ptr->name); 
       ptr = ptr->next) {

    if (ptr->onflg) 
      indent_level++;
  }

  if (indent_level > max_indent_level[mythread])
    max_indent_level[mythread] = indent_level;
    
  /* 
  ** If a new thing is being timed, add a new link and initialize 
  */

  if (ptr == NULL) {

    if ((ptr = (struct node *) malloc (sizeof (struct node))) == NULL)
      return (GPTerror ("GPTstart: malloc failed\n"));

    memset (ptr, 0, sizeof (struct node));
    ptr->indent_level = indent_level;
    ptr->next = NULL;

    if (timers[mythread] == NULL)
      timers[mythread] = ptr;
    else
      last[mythread]->next = ptr;

    last[mythread] = ptr;

    /* 
    ** Truncate input name if longer than MAX_CHARS characters 
    */

    numchars = MIN (strlen (name), MAX_CHARS);
    strncpy (ptr->name, name, numchars);
    ptr->name[numchars] = '\0';

  } else {

    /*
    ** If computed indentation level is different than before or was
    ** already ambiguous, reset to ambiguous flag value.  This will likely
    ** happen any time the thing being timed is called from more than 1
    ** branch in the call tree.
    */

    if (ptr->indent_level != indent_level) 
      ptr->indent_level = AMBIGUOUS;

    if (ptr->onflg)
      return GPTerror ("GPTstart thread %d: timer %s was already on: "
		       "not restarting.\n", mythread, ptr->name);
  }

  ptr->onflg = true;

  if (usrsysenabled)
    if (get_cpustamp (&ptr->last_utime, &ptr->last_stime) < 0)
      return GPTerror ("GPTstart: get_cpustamp error");

  /*
  ** The 2nd system timer call is used both for overhead estimation and
  ** the input timer
  */

  if (wallenabled) {

    gettimeofday (&tp2, NULL);
    ptr->last_wtime_sec  = tp2.tv_sec;
    ptr->last_wtime_usec = tp2.tv_usec;
    overhead[mythread] +=       (tp2.tv_sec  - tp1.tv_sec) + 
                          1.e-6*(tp2.tv_usec - tp1.tv_usec);
  }

  if (npcl > 0) {
    int n;
    int index;

    ret = PCLread (descr[mythread], i_pcl_result2, fp_pcl_result, npcl); 
    if (ret != PCL_SUCCESS)
      return GPTerror ("GPTstart: error from PCLread: %s\n", pclstr (ret));

    for (n = 0; n < npcl; n++) {
      ptr->last_pcl_result[n] = i_pcl_result2[n];
    }

    if (pcl_cyclesenabled) {
      index = pcl_cyclesindex;
      overhead_pcl[mythread] += i_pcl_result2[index] - i_pcl_result1[index];
    }
  } 

  return (0);
}

/*
** This stub should never actually be called
*/

#ifndef HAVE_PCL
int PCLread (PCL_DESCR_TYPE descr, PCL_CNT_TYPE *i, PCL_CNT_TYPE *j, int k)
{
  return GPTerror ("PCLread called when library not there\n");
}
#endif

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
  struct node *ptr;         /* linked list pointer */

  int mythread;             /* thread number for this process */
  int ret;                  /* return code */

  long usr;
  long sys;

  PCL_CNT_TYPE i_pcl_result1[PCL_COUNTER_MAX];     /* init. output fm PCLread */
  PCL_CNT_TYPE i_pcl_result2[PCL_COUNTER_MAX];     /* final output fm PCLread */
  PCL_FP_CNT_TYPE fp_pcl_result[PCL_COUNTER_MAX];  /* required by PCLread */

  if ( ! t_initialized)
    return GPTerror ("GPTstop: GPTinitialize has not been called\n");

  /*
  ** The 1st system timer call is used both for overhead estimation and
  ** the input timer
  */

  if (wallenabled)
    gettimeofday (&tp1, NULL);

  if (usrsysenabled && get_cpustamp (&usr, &sys) < 0)
    return GPTerror (NULL);

  if ((mythread = get_thread_num ()) < 0)
    return GPTerror ("GPTstop\n");

  if (npcl > 0) {
    ret = PCLread (descr[mythread], i_pcl_result1, fp_pcl_result, npcl);
    if (ret != PCL_SUCCESS)
      return GPTerror ("GPTstop: error from PCLread: %s\n", pclstr (ret));
  }
  
  for (ptr = timers[mythread]; ptr != NULL && ! STRMATCH (name, ptr->name); 
       ptr = ptr->next);

  if (ptr == NULL) 
    return GPTerror ("GPTstop: timer for %s had not been started.\n", name);

  if ( ! ptr->onflg )
    return GPTerror ("GPTstop: timer %s was already off.\n",ptr->name);

  ptr->onflg = false;
  ptr->count++;

  /*
  ** 1st timer stoppage: set max and min to computed values.  Otherwise apply
  ** max or min function
  */

  if (wallenabled) {

    delta_wtime_sec  = tp1.tv_sec  - ptr->last_wtime_sec;
    delta_wtime_usec = tp1.tv_usec - ptr->last_wtime_usec;
    delta_wtime      = delta_wtime_sec + 1.e-6*delta_wtime_usec;

    if (ptr->count == 1) {
      ptr->max_wtime = delta_wtime;
      ptr->min_wtime = delta_wtime;
      
    } else {
      
      ptr->max_wtime = MAX (ptr->max_wtime, delta_wtime);
      ptr->min_wtime = MIN (ptr->min_wtime, delta_wtime);
    }

    ptr->accum_wtime_sec  += delta_wtime_sec;
    ptr->accum_wtime_usec += delta_wtime_usec;

    /*
    ** Adjust accumulated wallclock values to guard against overflow in the
    ** microsecond accumulator.
    */

    if (ptr->accum_wtime_usec > 1000000) {
      ptr->accum_wtime_usec -= 1000000;
      ptr->accum_wtime_sec  += 1;
      
    } else if (ptr->accum_wtime_usec < -1000000) {
      
      ptr->accum_wtime_usec += 1000000;
      ptr->accum_wtime_sec  -= 1;
    }

    ptr->last_wtime_sec  = tp1.tv_sec;
    ptr->last_wtime_usec = tp1.tv_usec;

    /*
    ** 2nd system timer call is solely for overhead timing
    */

    gettimeofday (&tp2, NULL);
    overhead[mythread] +=       (tp2.tv_sec  - tp1.tv_sec) + 
                          1.e-6*(tp2.tv_usec - tp1.tv_usec);
  }

  if (usrsysenabled) {
    ptr->accum_utime += usr - ptr->last_utime;
    ptr->accum_stime += sys - ptr->last_stime;
    ptr->last_utime   = usr;
    ptr->last_stime   = sys;
  }

  if (npcl > 0) {
    int n;
    PCL_CNT_TYPE delta;
    int index;

    for (n = 0; n < npcl; n++) {
      delta = i_pcl_result1[n] - ptr->last_pcl_result[n];

      /*
      ** Accumulate results only for positive delta
      */

      if (delta < 0) 
	printf ("GPTstop: negative delta => probable counter overflow. "
		"Skipping accumulation this round\n"
		"%ld - %ld = %ld\n", (long) i_pcl_result1[n], 
		                     (long) ptr->last_pcl_result[n],
		                     (long) delta);
      else
	ptr->accum_pcl_result[n] += delta;

      ptr->last_pcl_result[n] = i_pcl_result1[n];
    }

    /*
    ** Overhead estimate.  Currently no check for negative delta
    */

    ret = PCLread (descr[mythread], i_pcl_result2, fp_pcl_result, npcl);
    if (ret != PCL_SUCCESS)
      return GPTerror ("GPTstop: error from PCLread: %s\n", t_pclstr (ret));

    if (pcl_cyclesenabled) {
      index = pcl_cyclesindex;
      overhead_pcl[mythread] += i_pcl_result2[index] - i_pcl_result1[index];
    }
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
  static long ticks_per_sec;
  static Boolean first = true;

  struct timeval tp;         /* argument to gettimeofday */
  struct tms buf;            /* argument to times */

  /*
  ** Not strictly thread-safe
  */

  if (first) {
    if ((ticks_per_sec = sysconf (_SC_CLK_TCK)) == -1)
      return GPTerror ("GPTstamp: token _SC_CLK_TCK is not defined\n");

    first = false;
  }

  *usr = 0;
  *sys = 0;

  if (usrsysenabled) {

    if (times (&buf) == -1)
      return GPTerror ("GPTstamp: times() failed. Timing bogus\n");

    *usr = buf.tms_utime / (double) ticks_per_sec;
    *sys = buf.tms_stime / (double) ticks_per_sec;

  } else {

    *usr = 0;
    *sys = 0;
  }

  gettimeofday (&tp, NULL);
  *wall = tp.tv_sec + 1.e-6*tp.tv_usec;

  return 0;
}

struct Stats {		   
  float usr;	   /* user CPU time */
  float sys;	   /* system CPU time */
  float usrsys;	   /* usr + sys */
  float elapse;	   /* elapsed time */
  float max_wtime; /* max elapsed wallclock time per call */
  float min_wtime; /* min elapsed wallclock time per call */

  long count;	   /* number of invocations of this timer */

  PCL_CNT_TYPE pcl_result[PCL_COUNTER_MAX];
};

/*
** GPTpr: print stats for all known timers to a file
**
** Input arguments:
**   procid: Designed for MPI jobs to give a unique output file name, 
**     normally the MPI logical process id.
**
** Return value: 0 (success) or -1 (failure)
*/


int GPTpr (int procid)
{
  static int mhz;          /* processor clock rate (from pcl library) */
  char outfile[11];        /* name of output file: timing.xxx */
			   
  int indent;              /* index over number indentation levels */
  int thread;              /* thread index */
  int ilstart;             /* index over indentation level */
  int n;

  double gttdcost;         /* cost of a single gettimeofday call */
  double deltat;           /* time difference between 2 calls to gettimeofday */
  struct Stats stats;      /* per timer stats */
  struct Stats threadsum;  /* timer sum over threads */
			   
  struct node *ptr, *tptr; /* linked list pointers */
			   
  FILE *fp;                /* output file pointer */

  struct timeval tp1, tp2; /* input to gettimeofday() */
  struct tms buf;          /* input to times() */

  if ( ! t_initialized)
    return GPTerror ("t_pr: t_initialize has not been called\n");

  if ((ticks_per_sec = sysconf (_SC_CLK_TCK)) == -1)
    return GPTerror ("t_pr: token _SC_CLK_TCK is not defined\n");

  /*
  ** Only allow the master thread to print stats
  */

  if (get_thread_num () != 0)
    return 0;

  sprintf (outfile, "%s%d\0","timing.", procid);

  if ((fp = fopen (outfile, "w")) == NULL)
    fp = stderr;

  fprintf (fp,"Procid = %d\n", procid);

  /*
  ** Estimate wallclock timer overhead: 4 times the cost of a call to gettimeofday
  ** (since each start/stop pair involves 4 such calls).
  */

  gettimeofday (&tp1, NULL);
  gettimeofday (&tp2, NULL);
  gttdcost = 1.e6*(tp2.tv_sec  - tp1.tv_sec) + (tp2.tv_usec - tp1.tv_usec);

  fprintf (fp, "Wallclock timer cost est.: %8.3g usec per start/stop pair\n", 
	   gttdcost*4.);

  /*
  ** CPU cost estimate: 2 times the cost of a call to times().  Subtract the
  ** cost of a single gettimeofday call to improve the estimate.
  */

  if (usrsysenabled) {

    gettimeofday (&tp1, NULL);
    (void) times (&buf);
    gettimeofday (&tp2, NULL);

    deltat = 1.e6*(tp2.tv_sec  - tp1.tv_sec) + (tp2.tv_usec - tp1.tv_usec);
    fprintf (fp, "CPU timer cost est.:       %8.3g usec per start/stop pair\n", 
	     2.*deltat - gttdcost);
  }
    
  fprintf (fp, "CPU accumulation interval is %g seconds\n",
           1./(float) ticks_per_sec);

#ifdef HAVE_PCL
  mhz = PCL_determine_mhz_rate();
  fprintf (fp, "Clock speed is %d MHz\n", mhz);
#endif

  for (thread = 0; thread < numthreads; thread++) {

    /*
    ** Only print heading for threads that have 1 or more items to report
    */

    if (timers[thread] == NULL) continue;
    fprintf (fp, "\nStats for thread %d:\n", thread);
#ifdef THREADED_PTHREADS
    fprintf (fp, "pthread id %d:\n", (int) threadid[thread]);
#endif

    print_header (fp, max_indent_level[thread]);

    for (ptr = timers[thread]; ptr != NULL; ptr = ptr->next) {
      if (ptr->onflg) {

	fprintf (fp, "Timer %s was not off.  No stats will be printed\n",
		 ptr->name);

      } else {

	fillstats (&stats, ptr);

	/*
	** Print stats indented.  If indent_level = AMBIGUOUS (a negative 
	** number) the name will be printed with no indentation.
	*/

	for (indent = 0; indent < ptr->indent_level; indent++)
	  fprintf (fp, "  ");
	fprintf (fp, "%-15s", ptr->name);

	/*
	** If indent_level = AMBIGUOUS (a negative number) we want to loop 
	** from 0
	*/

	ilstart = MAX (0, ptr->indent_level);
	for (indent = ilstart; indent < max_indent_level[thread]; indent++)
	  fprintf (fp, "  ");

	print_stats_line (fp, &stats);
      }
    }

    if (usrsysenabled || wallenabled)
      fprintf (fp, "\nTIMER OVERHEAD (wallclock seconds) = %12.6f\n", 
	       overhead[thread]);

    if (pcl_cyclesenabled)
      fprintf (fp, "TIMER OVERHEAD (cycles) = %12.6e\n", 
	       (double) overhead_pcl[thread]);
  }

  /*
  ** Print a vertical summary if data exist for more than 1 thread.  The "2"
  ** passed to print_header is so we'll get an extra 4 spaces of indentation
  ** due to the thread number appended to the timer name.
  */

  if (numthreads > 0 && timers[1] != NULL) {
    fprintf (fp, "\nSame stats sorted by timer with thread number appended:\n");
    print_header (fp, 2);

    /*
    ** Print stats for slave threads that match master
    */

    for (ptr = timers[0]; ptr != NULL; ptr = ptr->next) {

      char name[20];
      Boolean found = false;

      /*
      ** Don't bother printing summation stats when only the master thread
      ** invoked the timer
      */

      for (thread = 1; thread < numthreads; thread++)
	for (tptr = timers[thread]; tptr != NULL; tptr = tptr->next) {
	  if (STRMATCH (ptr->name, tptr->name))
	    found = true;
	}
      if ( ! found) continue;

      /*
      ** Initialize stats which sum over threads
      */

      memset (&threadsum, 0, sizeof (threadsum));

      if ( ! ptr->onflg) {
	fillstats (&stats, ptr);
	strcpy (name, ptr->name);
	strcat (name, ".0");
	fprintf (fp, "%-19s", name);
	print_stats_line (fp, &stats);
	threadsum = stats;
      }

      /*
      ** loop over slave threads, printing stats for each and accumulating
      ** sum over threads when the name matches
      */

      for (thread = 1; thread < numthreads; thread++) {
	for (tptr = timers[thread]; tptr != NULL; tptr = tptr->next) {
	  if (STRMATCH (ptr->name, tptr->name)) {
	    if ( ! tptr->onflg) {
	      char num[5];

	      fillstats (&stats, tptr);
	      strcpy (name, tptr->name);
	      sprintf (num, ".%-3d", thread);
	      strcat (name, num);
	      fprintf (fp, "%-19s", name);
	      print_stats_line (fp, &stats);

	      threadsum.usr      += stats.usr;
	      threadsum.sys      += stats.sys;
	      threadsum.usrsys   += stats.usrsys;
	      threadsum.elapse   += stats.elapse;
	      threadsum.max_wtime = MAX (threadsum.max_wtime, stats.max_wtime);
	      threadsum.min_wtime = MIN (threadsum.min_wtime, stats.min_wtime);
	      threadsum.count    += stats.count;

	      for (n = 0; n < npcl; n++)
		threadsum.pcl_result[n] += stats.pcl_result[n];
	    }
	    break; /* Go to the next thread */
	  }        /* if (STRMATCH (ptr->name, tptr->name) */
	}          /* loop thru linked list of timers for this thread */
      }            /* loop over slave threads */

      strcpy (name, ptr->name);
      strcat (name, ".sum");
      fprintf (fp, "%-19s", name);
      print_stats_line (fp, &threadsum);
      fprintf (fp, "\n");

    } /* loop through master timers */

    for (thread = 0; thread < numthreads; thread++) {
      if (usrsysenabled || wallenabled)
	fprintf (fp, "OVERHEAD.%-3d (wallclock seconds) = %12.6f\n", 
		 thread, overhead[thread]);

      if (pcl_cyclesenabled)
	fprintf (fp, "OVERHEAD.%-3d (cycles) = %12.6e\n", 
		 thread, (double) overhead_pcl[thread]);
    }

  } /* if (numthreads > 0 && timers[1] != NULL */

  return 0;
}

void fillstats (struct Stats *stats, struct node *ptr)
{
  int n;

  stats->usr       = ptr->accum_utime / (float) ticks_per_sec;
  stats->sys       = ptr->accum_stime / (float) ticks_per_sec;
  stats->usrsys    = stats->usr + stats->sys;
  stats->elapse    = ptr->accum_wtime_sec + 1.e-6 * ptr->accum_wtime_usec;
  stats->max_wtime = ptr->max_wtime;
  stats->min_wtime = ptr->min_wtime;
  stats->count     = ptr->count;

  for (n = 0; n < npcl; n++)
    stats->pcl_result[n] = ptr->accum_pcl_result[n];
}

void print_stats_line (FILE *fp, struct Stats *stats)
{
  int index;
  int n;
  long long cycles;
  long long instr;
  long long flops;
  long long loadstore;
  long long l2cache;
  long long jump;

  float mflops;
  float ipc;
  float memfp;

  fprintf (fp, "%9ld ", stats->count);

  if (usrsysenabled)
    fprintf (fp, "%9.3f %9.3f %9.3f ", stats->usr, stats->sys, stats->usrsys);

  if (wallenabled)
    fprintf (fp, "%9.3f %9.3f %9.3f ", stats->elapse, stats->max_wtime, 
	     stats->min_wtime);
  
  for (n = 0; n < nevent; n++) {
    if (event[n]->name > pcl_start && event[n]->name < pcl_end) {
      index = event[n]->index;
      if (stats->pcl_result[index] > 1.e6)
	fprintf (fp, "%9.3e ", (double) stats->pcl_result[index]);
      else
	fprintf (fp, "%9ld ", (long) stats->pcl_result[index]);
    }
  }

  fprintf (fp, "\n");
}

void print_header (FILE *fp, int indent_level)
{
  int i;
  int n;

  fprintf (fp, "Name           ");
  for (i = 0; i < indent_level; i++)
    fprintf (fp, "  ");
  fprintf (fp, "Called    ");

  if (usrsysenabled)
    fprintf (fp, "Usr       Sys       Usr+Sys   ");

  if (wallenabled)
    fprintf (fp, "Wallclock Max       Min       ");

  for (n = 0; n < nevent; n++)
    if (event[n]->name > pcl_start && event[n]->name <= pcl_end)
      fprintf (fp, event[n]->string);

  fprintf (fp, "\n");
}
