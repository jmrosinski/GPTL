/**
 * @file Main file contains most user-accessible GPTL functions.
 * @Author Jim Rosinski
 */

#include "config.h" // Must be first include.
#include "thread.h"

#ifdef HAVE_LIBMPI
#include <mpi.h>
#endif

#include <stdlib.h>        // malloc
#include <sys/time.h>      // gettimeofday
#include <sys/times.h>     // times
#include <unistd.h>        // gettimeofday, syscall
#include <stdio.h>
#include <string.h>        // memset, strcmp (via STRMATCH)
#include <ctype.h>         // isdigit

#ifdef HAVE_LIBRT
#include <time.h>
#endif

#ifdef _AIX
#include <sys/systemcfg.h>
#endif

#ifdef HAVE_LIBUNWIND
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#endif

#ifdef HAVE_BACKTRACE
#include <execinfo.h>
#endif

#include "private.h"
#include "gptl.h"

static Timer **timers = 0;             // linked list of timers
static Timer **last = 0;               // last element in list

static int depthlimit = 99999;         // max depth for timers (99999 is effectively infinite)
static volatile bool disabled = false; // Timers disabled?
static volatile bool initialized = false;        // GPTLinitialize has been called
#ifdef HAVE_PAPI
Entry GPTLeventlist[MAX_AUX];          // list of PAPI-based events to be counted
int GPTLnevents = 0;                   // number of PAPI events (init to 0)
#endif
static bool dousepapi = false;         // saves a function call if stays false
static bool verbose = false;           // output verbosity
static bool percent = false;           // print wallclock also as percent of 1st timers[0]
static bool dopr_preamble = true;      // whether to print preamble info
static bool dopr_threadsort = true;    // whether to print sorted thread stats
static bool dopr_multparent = true;    // whether to print multiple parent info
static bool dopr_collision = false;    // whether to print hash collision info
static bool dopr_memusage = false;     // whether to include memusage print when auto-profiling

static time_t ref_gettimeofday = -1;   // ref start point for gettimeofday
static time_t ref_clock_gettime = -1;  // ref start point for clock_gettime
#ifdef _AIX
static time_t ref_read_real_time = -1; // ref start point for read_real_time
#endif

typedef struct {
  const GPTLoption option;  // wall, cpu, etc.
  const char *str;          // descriptive string for printing
  bool enabled;             // flag
} Settings;

// Options, print strings, and default enable flags
static Settings cpustats =      {GPTLcpu,      "     usr       sys  usr+sys", false};
static Settings wallstats =     {GPTLwall,     "     Wall      max      min", true };
static Settings overheadstats = {GPTLoverhead, "   selfOH parentOH"         , true };

static Hashentry **hashtable;    // table of entries
static long ticks_per_sec;       // clock ticks per second
static Timer ***callstack;       // call stack
static Nofalse *stackidx;        // index into callstack:

static GPTLMethod method = GPTLmost_frequent;  // default parent/child printing mechanism

typedef struct {
  int max_depth;
  int max_namelen;
  int max_chars2pr;
} Outputfmt;

// Local function prototypes
static inline int preamble_start (int *, const char *);
static inline int preamble_stop (int *, double *, long *, long *, const char *);
static int get_longest_omp_namelen (void);
static int get_outputfmt (const Timer *, const int, const int, Outputfmt *);
static void fill_output (int, int, int, Outputfmt *);
static void printstats (const Timer *, FILE *, int, int, bool, double, double, const Outputfmt);
static void print_titles (int, FILE *, Outputfmt *);
static void add (Timer *, const Timer *);
static void print_multparentinfo (FILE *, Timer *);
static inline int get_cpustamp (long *, long *);
static int newchild (Timer *, Timer *);
static int get_max_namelen (Timer *);
static int is_descendant (const Timer *, const Timer *);
static int is_onlist (const Timer *, const Timer *);
static char *methodstr (GPTLMethod);
static void print_callstack (int, const char *);
static int get_symnam (void *, char **);

  // These are the (possibly) supported underlying wallclock timers
#ifdef HAVE_NANOTIME
static int init_nanotime (void);
static inline double utr_nanotime (void);
#endif
 
#ifdef HAVE_LIBMPI
static int init_mpiwtime (void);
static inline double utr_mpiwtime (void);
#endif
  
#ifdef _AIX
static int init_read_real_time (void);
static inline double utr_read_real_time (void);
#endif

#ifdef HAVE_LIBRT
static int init_clock_gettime (void);
static inline double utr_clock_gettime (void);
#endif
  
static int init_gettimeofday (void);
static inline double utr_gettimeofday (void);

static int init_placebo (void);
static inline double utr_placebo (void);

static inline unsigned int genhashidx (const char *);
static inline Timer *getentry_instr (const Hashentry *, void *, unsigned int *);
static inline Timer *getentry (const Hashentry *, const char *, unsigned int);
static void printself_andchildren (const Timer *, FILE *, int, int, double, double, Outputfmt);
static inline int update_parent_info (Timer *, Timer **, int);
static inline int update_stats (Timer *, const double, const long, const long, const int);
static int update_ll_hash (Timer *, int, unsigned int);
static inline int update_ptr (Timer *, const int);
static int construct_tree (Timer *, GPTLMethod);
static inline void set_fp_procsiz (void);
static void check_memusage (const char *, const char *);
static void translate_truncated_names (int, FILE *);

bool GPTLonlypr_rank0 = false;    // flag says only print from MPI rank 0 (default false)

typedef struct {
  const GPTLFuncoption option;
  double (*func)(void);
  int (*funcinit)(void);
  const char *name;
} Funcentry;

static Funcentry funclist[] = {
  {GPTLgettimeofday,   utr_gettimeofday,   init_gettimeofday,  "gettimeofday"},
#ifdef HAVE_NANOTIME
  {GPTLnanotime,       utr_nanotime,       init_nanotime,      "nanotime"},
#endif
#ifdef HAVE_LIBMPI
  {GPTLmpiwtime,       utr_mpiwtime,       init_mpiwtime,      "MPI_Wtime"},
#endif
#ifdef HAVE_LIBRT
  {GPTLclockgettime,   utr_clock_gettime,  init_clock_gettime, "clock_gettime"},
#endif
#ifdef _AIX
  {GPTLread_real_time, utr_read_real_time, init_read_real_time,"read_real_time"},     // AIX only
#endif
  {GPTLplacebo,        utr_placebo,        init_placebo,       "placebo"}      // does nothing
};
static const int nfuncentries = sizeof (funclist) / sizeof (Funcentry);

static double (*ptr2wtimefunc)() = 0;    // init to invalid
static char unknown[] = "unknown";

#ifdef HAVE_NANOTIME
static float cpumhz = -1.;               // init to bad value
static double cyc2sec = -1;              // init to bad value
static inline long long nanotime (void); // read counter (assembler)
static float get_clockfreq (void);       // cycles/sec
static char *clock_source = unknown;     // where clock found
#endif
static int funcidx = 0;                  // default timer is gettimeofday

#define DEFAULT_TABLE_SIZE 1023
static int tablesize = DEFAULT_TABLE_SIZE;  // per-thread size of hash table (settable parameter)
static int tablesizem1 = DEFAULT_TABLE_SIZE - 1;

static float rssmax = 0;                 // max rss of the process
static bool imperfect_nest;              // e.g. start(A),start(B),stop(A)
static const int indent_chars = 2;       // Number of chars to indent
static FILE *fp_procsiz = 0;             // process size file pointer: init to 0 to use stderr

// VERBOSE is a debugging ifdef local to the rest of this file
#undef VERBOSE
// APPEND_ADDRESS is an autoprofiling debugging ifdef that will append function address to name
#undef APPEND_ADDRESS

/**
 * Set option value
 *
 * @param option option to be set
 * @param val value to which option should be set (nonzero=true, zero=false)
 *
 * @return: 0 (success) or GPTLerror (failure)
 */
int GPTLsetoption (const int option, const int val)
{
  static const char *thisfunc = "GPTLsetoption";

  if (initialized)
    return GPTLerror ("%s: must be called BEFORE GPTLinitialize\n", thisfunc);

  if (option == GPTLabort_on_error) {
    GPTLset_abort_on_error ((bool) val);
    if (verbose)
      printf ("%s: boolean abort_on_error = %d\n", thisfunc, val);
    return 0;
  }

  switch (option) {
  case GPTLcpu:
#ifdef HAVE_TIMES
    cpustats.enabled = (bool) val; 
    if (verbose)
      printf ("%s: cpustats = %d\n", thisfunc, val);
#else
    if (val)
      return GPTLerror ("%s: times() not available\n", thisfunc);
#endif
    return 0;
  case GPTLwall:     
    wallstats.enabled = (bool) val; 
    if (verbose)
      printf ("%s: boolean wallstats = %d\n", thisfunc, val);
    return 0;
  case GPTLoverhead: 
    overheadstats.enabled = (bool) val; 
    if (verbose)
      printf ("%s: boolean overheadstats = %d\n", thisfunc, val);
    return 0;
  case GPTLdepthlimit: 
    depthlimit = val; 
    if (verbose)
      printf ("%s: depthlimit = %d\n", thisfunc, val);
    return 0;
  case GPTLverbose: 
    verbose = (bool) val; 
#ifdef HAVE_PAPI
    (void) GPTL_PAPIsetoption (GPTLverbose, val);
#endif
    if (verbose)
      printf ("%s: boolean verbose = %d\n", thisfunc, val);
    return 0;
  case GPTLpercent: 
    percent = (bool) val; 
    if (verbose)
      printf ("%s: boolean percent = %d\n", thisfunc, val);
    return 0;
  case GPTLdopr_preamble: 
    dopr_preamble = (bool) val; 
    if (verbose)
      printf ("%s: boolean dopr_preamble = %d\n", thisfunc, val);
    return 0;
  case GPTLdopr_threadsort: 
    dopr_threadsort = (bool) val; 
    if (verbose)
      printf ("%s: boolean dopr_threadsort = %d\n", thisfunc, val);
    return 0;
  case GPTLdopr_multparent: 
    dopr_multparent = (bool) val; 
    if (verbose)
      printf ("%s: boolean dopr_multparent = %d\n", thisfunc, val);
    return 0;
  case GPTLdopr_collision: 
    dopr_collision = (bool) val; 
    if (verbose)
      printf ("%s: boolean dopr_collision = %d\n", thisfunc, val);
    return 0;
  case GPTLdopr_memusage: 
    dopr_memusage = (bool) val; 
    if (verbose)
      printf ("%s: boolean dopr_memusage = %d\n", thisfunc, val);
    return 0;
  case GPTLprint_method:
    method = (GPTLMethod) val; 
    if (verbose)
      printf ("%s: print_method = %s\n", thisfunc, methodstr (method));
    return 0;
  case GPTLtablesize:
    if (val < 1)
      return GPTLerror ("%s: tablesize must be positive. %d is invalid\n", thisfunc, val);
    tablesize = val;
    tablesizem1 = val - 1;
    if (verbose)
      printf ("%s: tablesize = %d\n", thisfunc, tablesize);
    return 0;
  case GPTLsync_mpi:
#ifdef ENABLE_PMPI
    if (GPTLpmpi_setoption (option, val) != 0)
      fprintf (stderr, "%s: GPTLpmpi_setoption failure\n", thisfunc);
    if (verbose)
      printf ("%s: boolean sync_mpi = %d\n", thisfunc, val);
#else
    fprintf (stderr, "%s: option GPTLsync_mpi requires MPI\n", thisfunc);
#endif
    return 0;
  case GPTLmaxthreads:
    if (val < 1)
      return GPTLerror ("%s: GPTLmaxthreads must be positive. %d is invalid\n", thisfunc, val);

    GPTLmax_threads = val;
    return 0;
  case GPTLonlyprint_rank0:
    GPTLonlypr_rank0 = (bool) val; 
    if (verbose)
      printf ("%s: onlypr_rank0 = %d\n", thisfunc, val);
    return 0;
    
  case GPTLmultiplex:
    // Allow GPTLmultiplex to fall through because it will be handled by GPTL_PAPIsetoption()
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
  return GPTLerror ("%s: failure to enable option %d\n", thisfunc, option);
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
  int i;  // index over number of underlying timer
  static const char *thisfunc = "GPTLsetutr";

  if (initialized)
    return GPTLerror ("%s: must be called BEFORE GPTLinitialize\n", thisfunc);

  for (i = 0; i < nfuncentries; i++) {
    if (option == (int) funclist[i].option) {
      if (verbose)
        printf ("%s: underlying wallclock timer = %s\n", thisfunc, funclist[i].name);
      funcidx = i;

      // Return an error condition if the function is not available.
      // OK for the user code to ignore: GPTLinitialize() will reset to gettimeofday
      if ((*funclist[i].funcinit)() < 0)
        return GPTLerror ("%s: utr=%s not available or doesn't work\n", thisfunc, funclist[i].name);
      else
        return 0;
    }
  }
  return GPTLerror ("%s: unknown option %d\n", thisfunc, option);
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
  int i;
  int t;
  double t1, t2;  // returned from underlying timer
  static const char *thisfunc = "GPTLinitialize";

  if (initialized)
    return GPTLerror ("%s: has already been called\n", thisfunc);

  if (GPTLthreadinit () < 0)
    return GPTLerror ("%s: bad return from GPTLthreadinit\n", thisfunc);

  if ((ticks_per_sec = sysconf (_SC_CLK_TCK)) == -1)
    return GPTLerror ("%s: failure from sysconf (_SC_CLK_TCK)\n", thisfunc);

  // Allocate space for global arrays
  callstack       = (Timer ***)    GPTLallocate (GPTLmax_threads * sizeof (Timer **), thisfunc);
  stackidx        = (Nofalse *)    GPTLallocate (GPTLmax_threads * sizeof (Nofalse), thisfunc);
  timers          = (Timer **)     GPTLallocate (GPTLmax_threads * sizeof (Timer *), thisfunc);
  last            = (Timer **)     GPTLallocate (GPTLmax_threads * sizeof (Timer *), thisfunc);
  hashtable       = (Hashentry **) GPTLallocate (GPTLmax_threads * sizeof (Hashentry *), thisfunc);

  // Initialize array values
  for (t = 0; t < GPTLmax_threads; t++) {
    callstack[t] = (Timer **) GPTLallocate (MAX_STACK * sizeof (Timer *), thisfunc);
    hashtable[t] = (Hashentry *) GPTLallocate (tablesize * sizeof (Hashentry), thisfunc);
    for (i = 0; i < tablesize; i++) {
      hashtable[t][i].nument = 0;
      hashtable[t][i].entries = 0;
    }

    // Make a timer "GPTL_ROOT" to ensure no orphans, and to simplify printing
    timers[t] = (Timer *) GPTLallocate (sizeof (Timer), thisfunc);
    memset (timers[t], 0, sizeof (Timer));
    strcpy (timers[t]->name, "GPTL_ROOT");
    timers[t]->onflg = true;
    last[t] = timers[t];

    stackidx[t].val = 0;
    callstack[t][0] = timers[t];
    for (i = 1; i < MAX_STACK; i++)
      callstack[t][i] = 0;
  }

#ifdef HAVE_PAPI
  if (GPTL_PAPIinitialize (GPTLmax_threads, verbose, &GPTLnevents, GPTLeventlist) < 0)
    return GPTLerror ("%s: Failure from GPTL_PAPIinitialize\n", thisfunc);
#endif

  // Call init routine for underlying timing routine
  if ((*funclist[funcidx].funcinit)() < 0) {
    fprintf (stderr, "%s: Failure initializing %s. Reverting underlying timer to %s\n", 
             thisfunc, funclist[funcidx].name, funclist[0].name);
    funcidx = 0;
  }

  ptr2wtimefunc = funclist[funcidx].func;

  if (verbose) {
    t1 = (*ptr2wtimefunc) ();
    t2 = (*ptr2wtimefunc) ();
    if (t1 > t2)
      fprintf (stderr, "%s: negative delta-t=%g\n", thisfunc, t2-t1);
    printf ("Per call overhead est. t2-t1=%g should be near zero\n", t2-t1);
    printf ("Underlying wallclock timing routine is %s\n", funclist[funcidx].name);
  }

  imperfect_nest = false;
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
  int t;
  int n;
  Timer *ptr, *ptrnext;
  static const char *thisfunc = "GPTLfinalize";

  if ( ! initialized)
    return GPTLerror ("%s: initialization was not completed\n", thisfunc);

  for (t = 0; t < GPTLmax_threads; ++t) {
    for (n = 0; n < tablesize; ++n) {
      if (hashtable[t][n].nument > 0)
        free (hashtable[t][n].entries);
    }
    free (hashtable[t]);
    hashtable[t] = NULL;
    free (callstack[t]);
    for (ptr = timers[t]; ptr; ptr = ptrnext) {
      ptrnext = ptr->next;
      if (ptr->nparent > 0) {
        free (ptr->parent);
        free (ptr->parent_count);
      }
      if (ptr->nchildren > 0)
        free (ptr->children);
      free (ptr);
    }
  }

  free (callstack);
  free (stackidx);
  free (timers);
  free (last);
  free (hashtable);

  GPTLthreadfinalize ();
  GPTLreset_errors ();

#ifdef HAVE_PAPI
  GPTL_PAPIfinalize (GPTLmax_threads);
#endif

  // Reset initial values
  timers = 0;
  last = 0;
  GPTLnthreads = -1;
#ifdef UNDERLYING_PTHREADS
  GPTLmax_threads = MAX_THREADS;
#else
  GPTLmax_threads = -1;
#endif
  depthlimit = 99999;
  disabled = false;
  initialized = false;
  dousepapi = false;
  verbose = false;
  percent = false;
  dopr_preamble = true;
  dopr_threadsort = true;
  dopr_multparent = true;
  dopr_collision = false;
  ref_gettimeofday = -1;
  ref_clock_gettime = -1;
#ifdef _AIX
  ref_read_real_time = -1;
#endif
  funcidx = 0;
#ifdef HAVE_NANOTIME
  cpumhz= 0;
  cyc2sec = -1;
#endif
  tablesize = DEFAULT_TABLE_SIZE;
  tablesizem1 = tablesize - 1;

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
int GPTLstart (const char *name)
{
  Timer *ptr;
  int t;
  int ret;
  int numchars;
  unsigned int indx; // index into hash table
  static const char *thisfunc = "GPTLstart";
  
  ret = preamble_start (&t, name);
  if (ret == DONE)
    return 0;
  else if (ret != 0)
    return ret;
  
  // ptr will point to the requested timer in the current list, or NULL if this is a new entry
  indx = genhashidx (name);
  ptr = getentry (hashtable[t], name, indx);

  /* 
  ** Recursion => increment depth in recursion and return.  We need to return 
  ** because we don't want to restart the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */
  if (ptr && ptr->onflg) {
    ++ptr->recurselvl;
    return 0;
  }

  // Increment stackidx[t] unconditionally. This is necessary to ensure the correct
  // behavior when GPTLstop decrements stackidx[t] unconditionally.
  if (++stackidx[t].val > MAX_STACK-1)
    return GPTLerror ("%s: stack too big: NOT starting timer for %s\n", thisfunc, name);

  if ( ! ptr) {   // Add a new entry and initialize. longname only needed for auto-profiling
    ptr = (Timer *) GPTLallocate (sizeof (Timer), thisfunc);
    memset (ptr, 0, sizeof (Timer));

    numchars = MIN (strlen (name), MAX_CHARS);
    strncpy (ptr->name, name, numchars);
    ptr->name[numchars] = '\0';

    if (update_ll_hash (ptr, t, indx) != 0)
      return GPTLerror ("%s: update_ll_hash error\n", thisfunc);
  }

  if (update_parent_info (ptr, callstack[t], stackidx[t].val) != 0)
    return GPTLerror ("%s: update_parent_info error\n", thisfunc);

  if (update_ptr (ptr, t) != 0)
    return GPTLerror ("%s: update_ptr error\n", thisfunc);

  if (dopr_memusage && t == 0)
    check_memusage ("Begin", ptr->name);

  return 0;
}

// preamble_start: Do the things common to GPTLstart* routines
static inline int preamble_start (int *t, const char *name)
{
  static const char *thisfunc = "preamble_start";
  if (disabled)
    return DONE;

  // Only print error message for manual instrumentation: too hard to ensure
  // GPTLinitialize() has been called for auto-instrumented code
  if ( ! initialized)
    return GPTLerror ("%s timername=%s: GPTLinitialize has not been called\n", thisfunc, name);
  
  if ((*t = GPTLget_thread_num ()) < 0)
    return GPTLerror ("%s: bad return from GPTLget_thread_num\n", thisfunc);

  // If current depth exceeds a user-specified limit for print, just
  // increment and tell caller to return immediately (DONTSTART)
  if (stackidx[*t].val >= depthlimit) {
    ++stackidx[*t].val;
    return DONE;
  }
  return 0;
}

/*
** GPTLinit_handle: Initialize a handle for further use by GPTLstart_handle() and GPTLstop_handle()
**
** Input arguments:
**   name: timer name
**
** Output arguments:
**   handle: hash value corresponding to "name"
**
** Return value: 0 (success) or GPTLerror (failure)
*/
int GPTLinit_handle (const char *name, int *handle)
{
  if (disabled)
    return 0;

  *handle = (int) genhashidx (name);
  return 0;
}

/*
** GPTLstart_handle: start a timer based on a handle
**
** Input arguments:
**   name: timer name (required when on input, handle=0)
**   handle: pointer to timer matching "name"
**
** Return value: 0 (success) or GPTLerror (failure)
*/
int GPTLstart_handle (const char *name, int *handle)
{
  Timer *ptr;
  int t;
  int ret;
  int numchars;
  static const char *thisfunc = "GPTLstart_handle";

  ret = preamble_start (&t, name);
  if (ret == DONE)
    return 0;
  else if (ret != 0)
    return ret;

  /*
  ** If handle is zero on input, generate the hash entry and return it to the user.
  ** Otherwise assume it's a previously generated hash index passed in by the user.
  ** Don't need a critical section here--worst case multiple threads will generate the
  ** same handle and store to the same memory location, and this will only happen once.
  */
  if (*handle == 0) {
    *handle = (int) genhashidx (name);
#ifdef VERBOSE
    printf ("%s: name=%s thread %d generated handle=%d\n", thisfunc, name, t, *handle);
#endif
  } else if ((unsigned int) *handle > tablesizem1) {
    return GPTLerror ("%s: Bad input handle=%u exceeds tablesizem1=%d\n", 
		      thisfunc, (unsigned int) *handle, tablesizem1);
  }

  ptr = getentry (hashtable[t], name, (unsigned int) *handle);
  
  /* 
  ** Recursion => increment depth in recursion and return.  We need to return 
  ** because we don't want to restart the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */
  if (ptr && ptr->onflg) {
    ++ptr->recurselvl;
    return 0;
  }

  // Increment stackidx[t] unconditionally. This is necessary to ensure the correct
  // behavior when GPTLstop decrements stackidx[t] unconditionally.
  if (++stackidx[t].val > MAX_STACK-1)
    return GPTLerror ("%s: stack too big: NOT starting timer for %s\n", thisfunc, name);

  if ( ! ptr) { // Add a new entry and initialize
    ptr = (Timer *) GPTLallocate (sizeof (Timer), thisfunc);
    memset (ptr, 0, sizeof (Timer));

    numchars = MIN (strlen (name), MAX_CHARS);
    strncpy (ptr->name, name, numchars);
    ptr->name[numchars] = '\0';

    if (update_ll_hash (ptr, t, (unsigned int) *handle) != 0)
      return GPTLerror ("%s: update_ll_hash error\n", thisfunc);
  }

  if (update_parent_info (ptr, callstack[t], stackidx[t].val) != 0)
    return GPTLerror ("%s: update_parent_info error\n", thisfunc);

  if (update_ptr (ptr, t) != 0)
    return GPTLerror ("%s: update_ptr error\n", thisfunc);

  if (dopr_memusage && t == 0)
    check_memusage ("Begin", ptr->name);

  return (0);
}

/*
** update_ll_hash: Update linked list and hash table.
**                 Called by all GPTLstart* routines when there is a new entry
**
** Input arguments:
**   ptr:  pointer to timer
**   t:    thread index
**   indx: hash index
**
** Return value: 0 (success) or GPTLerror (failure)
*/
static int update_ll_hash (Timer *ptr, int t, unsigned int indx)
{
  int nument;      // number of entries (> 0 means collision)
  Timer **eptr;    // for realloc

  last[t]->next = ptr;
  last[t] = ptr;
  ++hashtable[t][indx].nument;
  nument = hashtable[t][indx].nument;
  
  eptr = (Timer **) realloc (hashtable[t][indx].entries, nument * sizeof (Timer *));
  if ( ! eptr)
    return GPTLerror ("update_ll_hash: realloc error\n");

  hashtable[t][indx].entries           = eptr;
  hashtable[t][indx].entries[nument-1] = ptr;

  return 0;
}

/*
** update_ptr: Update timer contents. Called by GPTLstart, GPTLstart_handle, and
**             __cyg_profile_func_enter
**
** Input arguments:
**   ptr:  pointer to timer
**   t:    thread index
**
** Return value: 0 (success) or GPTLerror (failure)
*/
static inline int update_ptr (Timer *ptr, const int t)
{
  ptr->onflg = true;

  if (cpustats.enabled && get_cpustamp (&ptr->cpu.last_utime, &ptr->cpu.last_stime) < 0)
    return GPTLerror ("update_ptr: get_cpustamp error");
  
  if (wallstats.enabled) {
    double tp2 = (*ptr2wtimefunc) ();
    tp2 = (*ptr2wtimefunc) ();
    ptr->wall.last = tp2;
  }

#ifdef HAVE_PAPI
  if (dousepapi && GPTL_PAPIstart (t, &ptr->aux) < 0)
    return GPTLerror ("update_ptr: error from GPTL_PAPIstart\n");
#endif
  return 0;
}

/*
** update_parent_info: update info about parent, and in the parent about this child
**                     Called by all GPTLstart* routines
**
** Arguments:
**   ptr:  pointer to timer
**   callstackt: callstack for this thread
**   stackidxt:  stack index for this thread
**
** Return value: 0 (success) or GPTLerror (failure)
*/
static inline int update_parent_info (Timer *ptr, Timer **callstackt, int stackidxt)
{
  int n;             // loop index through known parents
  Timer *pptr;       // pointer to parent in callstack
  Timer **pptrtmp;   // for realloc parent pointer array
  int nparent;       // number of parents
  int *parent_count; // number of times parent invoked this child
  static const char *thisfunc = "update_parent_info";

  if ( ! ptr )
    return -1;

  if (stackidxt < 0)
    return GPTLerror ("%s: called with negative stackidx\n", thisfunc);

  callstackt[stackidxt] = ptr;

  // Bump orphan count if the region has no parent (should never happen since "GPTL_ROOT" added)
  if (stackidxt == 0) {
    ++ptr->norphan;
    return 0;
  }

  pptr = callstackt[stackidxt-1];

  // If this parent occurred before, bump its count
  for (n = 0; n < ptr->nparent; ++n) {
    if (ptr->parent[n] == pptr) {
      ++ptr->parent_count[n];
      break;
    }
  }

  // If this is a new parent, update info
  if (n == ptr->nparent) {
    ++ptr->nparent;
    nparent = ptr->nparent;
    pptrtmp = (Timer **) realloc (ptr->parent, nparent * sizeof (Timer *));
    if ( ! pptrtmp)
      return GPTLerror ("%s: realloc error pptrtmp nparent=%d\n", thisfunc, nparent);

    ptr->parent = pptrtmp;
    ptr->parent[nparent-1] = pptr;
    parent_count = (int *) realloc (ptr->parent_count, nparent * sizeof (int));
    if ( ! parent_count)
      return GPTLerror ("%s: realloc error parent_count nparent=%d\n", thisfunc, nparent);

    ptr->parent_count = parent_count;
    ptr->parent_count[nparent-1] = 1;
  }

  return 0;
}

/*
** GPTLstop: stop a timer
**
** Input arguments:
**   name: timer name
**
** Return value: 0 (success) or -1 (failure)
*/
int GPTLstop (const char *name)
{
  double tp1 = 0.0;          // wallclock time stamp
  Timer *ptr;
  int t;
  int ret;
  unsigned int indx;         // hash indexx
  long usr = 0;              // user time (returned from get_cpustamp)
  long sys = 0;              // system time (returned from get_cpustamp)
  static const char *thisfunc = "GPTLstop";

  ret = preamble_stop (&t, &tp1, &usr, &sys, name);
  if (ret == DONE)
    return 0;
  else if (ret != 0)
    return ret;
       
  indx = genhashidx (name);
  if (! (ptr = getentry (hashtable[t], name, indx)))
    return GPTLerror ("%s thread %d: timer for %s had not been started.\n", thisfunc, t, name);

  if ( ! ptr->onflg )
    return GPTLerror ("%s: timer %s was already off.\n", thisfunc, ptr->name);

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
  }

  if (update_stats (ptr, tp1, usr, sys, t) != 0)
    return GPTLerror ("%s: error from update_stats\n", thisfunc);

  if (dopr_memusage && t == 0)
    check_memusage ("End", ptr->name);

  return 0;
}

static inline int preamble_stop (int *t, double *tp1, long *usr, long *sys, const char *name)
{
  static const char *thisfunc = "preamble_stop";

  if (disabled)
    return DONE;

  if ( ! initialized)
    return GPTLerror ("%s timername=%s: GPTLinitialize has not been called\n", thisfunc, name);

  // Get the wallclock timestamp
  if (wallstats.enabled) {
    *tp1 = (*ptr2wtimefunc) ();
  }

  if (cpustats.enabled && get_cpustamp (usr, sys) < 0)
    return GPTLerror ("%s: get_cpustamp error", name);

  if ((*t = GPTLget_thread_num ()) < 0)
    return GPTLerror ("%s: bad return from GPTLget_thread_num\n", name);

  // If current depth exceeds a user-specified limit for print, just decrement and return
  if (stackidx[*t].val > depthlimit) {
    --stackidx[*t].val;
    return DONE;
  }
  return 0;
}

/*
** GPTLstop_handle: stop a timer based on a handle
**
** Input arguments:
**   name: timer name (used only for diagnostics)
**   handle: pointer to timer
**
** Return value: 0 (success) or -1 (failure)
*/
int GPTLstop_handle (const char *name, int *handle)
{
  double tp1 = 0.0;          // wallclock time stamp
  Timer *ptr;
  int t;
  int ret;
  long usr = 0;              // user time (returned from get_cpustamp)
  long sys = 0;              // system time (returned from get_cpustamp)
  unsigned int indx;         // index into hash table
  static const char *thisfunc = "GPTLstop_handle";

  ret = preamble_stop (&t, &tp1, &usr, &sys, thisfunc);
  if (ret == DONE)
    return 0;
  else if (ret != 0)
    return ret;
       
  indx = (unsigned int) *handle;
  if (indx == 0 || indx > tablesizem1) 
    return GPTLerror ("%s: bad input handle=%u for timer %s.\n", thisfunc, indx, name);
  
  if ( ! (ptr = getentry (hashtable[t], name, indx)))
    return GPTLerror ("%s: handle=%u has not been set for timer %s.\n", 
		      thisfunc, indx, name);

  if ( ! ptr->onflg )
    return GPTLerror ("%s: timer %s was already off.\n", thisfunc, ptr->name);

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
  }

  if (update_stats (ptr, tp1, usr, sys, t) != 0)
    return GPTLerror ("%s: error from update_stats\n", thisfunc);

  if (dopr_memusage && t == 0)
    check_memusage ("End", ptr->name);

  return 0;
}

/*
** update_stats: update stats inside ptr. Called by GPTLstop, GPTLstop_handle
**
** Input arguments:
**   ptr: pointer to timer
**   tp1: input time stamp
**   usr: user time
**   sys: system time
**   t: thread index
**
** Return value: 0 (success) or GPTLerror (failure)
*/
static inline int update_stats (Timer *ptr, const double tp1, const long usr, const long sys,
                                const int t)
{
  double delta;      // wallclock time difference
  int bidx;          // bottom of call stack
  Timer *bptr;       // pointer to last entry in call stack
  static const char *thisfunc = "update_stats";

  ptr->onflg = false;

#ifdef HAVE_PAPI
  if (dousepapi && GPTL_PAPIstop (t, &ptr->aux) < 0)
    return GPTLerror ("%s: error from GPTL_PAPIstop\n", thisfunc);
#endif

  if (wallstats.enabled) {
    delta = tp1 - ptr->wall.last;
    ptr->wall.accum += delta;
    ptr->wall.latest = delta;

    if (delta < 0.)
      fprintf (stderr, "GPTL: %s: negative delta=%g\n", thisfunc, delta);

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

  // Verify that the timer being stopped is at the bottom of the call stack
  if ( ! imperfect_nest) {
    char *name;        //  found name
    char *bname;       //  expected name

    bidx = stackidx[t].val;
    bptr = callstack[t][bidx];
    if (ptr != bptr) {
      imperfect_nest = true;
      if (ptr->longname)
	name = ptr->longname;
      else
	name = ptr->name;
      
      // Print to stderr as well due to debugging importance, and warn/error have limits on the
      // number of calls that can be made
      fprintf (stderr, "%s: Imperfect nest detected: Got timer %s\n", thisfunc, name);      
      GPTLwarn ("%s: Imperfect nest detected: Got timer %s\n", thisfunc, name);
      if (bptr) {
	if (bptr->longname)
	  bname = bptr->longname;
	else
	  bname = bptr->name;
	fprintf (stderr, "Expected btm of call stack %s\n", bname);
      } else {
	// Sometimes imperfect_nest can cause bptr to be NULL
	fprintf (stderr, "Expected btm of call stack but bptr is NULL\n");
      }
      //      print_callstack (t, thisfunc);
    }
  }

  --stackidx[t].val;           // Pop the callstack
  if (stackidx[t].val < -1) {
    stackidx[t].val = -1;
    return GPTLerror ("%s: tree depth has become negative.\n", thisfunc);
  }
  return 0;
}

// GPTLenable: enable timers 
int GPTLenable (void)
{
  disabled = false;
  return (0);
}

// GPTLdisable: disable timers
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
  struct tms buf;            // returned from times()

  if ( ! initialized)
    return GPTLerror ("GPTLstamp: GPTLinitialize has not been called\n");

#ifdef HAVE_TIMES
  *usr = 0;
  *sys = 0;

  if (times (&buf) == -1)
    return GPTLerror ("GPTLstamp: times() failed. Results bogus\n");

  *usr = buf.tms_utime / (double) ticks_per_sec;
  *sys = buf.tms_stime / (double) ticks_per_sec;
#endif
  *wall = (*ptr2wtimefunc) ();
  return 0;
}

// GPTLreset: reset all timers to 0
// Return value: 0 (success) or GPTLerror (failure)
int GPTLreset (void)
{
  int t;
  Timer *ptr;
  static const char *thisfunc = "GPTLreset";

  if ( ! initialized)
    return GPTLerror ("%s: GPTLinitialize has not been called\n", thisfunc);

  for (t = 0; t < GPTLnthreads; t++) {
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
    printf ("%s: accumulators for all timers set to zero\n", thisfunc);

  return 0;
}

// GPTLreset_timer: reset a timer to 0
// Return value: 0 (success) or GPTLerror (failure)
int GPTLreset_timer (char *name)
{
  int t;
  Timer *ptr;
  unsigned int indx; // hash index
  static const char *thisfunc = "GPTLreset_timer";

  if ( ! initialized)
    return GPTLerror ("%s: GPTLinitialize has not been called\n", thisfunc);

  if (GPTLget_thread_num () != 0)
    return GPTLerror ("%s: Must be called by the master thread\n", thisfunc);

  indx = genhashidx (name);
  for (t = 0; t < GPTLnthreads; ++t) {
    ptr = getentry (hashtable[t], name, indx);
    if (ptr) {
      ptr->onflg = false;
      ptr->count = 0;
      memset (&ptr->wall, 0, sizeof (ptr->wall));
      memset (&ptr->cpu, 0, sizeof (ptr->cpu));
#ifdef HAVE_PAPI
      memset (&ptr->aux, 0, sizeof (ptr->aux));
#endif
    }
  }
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
int GPTLpr (const int id) // output file will be named "timing.<id>" or stderr if negative or huge
{
  char outfile[14];       // name of output file: timing.xxxxxx
  static const char *thisfunc = "GPTLpr";

  // Not great hack to force output to stderr: input a negative or huge number
  if (id < 0 || id > 999999) {
    GPTLnote ("%s id=%d means output will be written to stderr\n", thisfunc, id);
    sprintf (outfile, "stderr");
  } else {
    sprintf (outfile, "timing.%6.6d", id);
  }

  if (GPTLpr_file (outfile) != 0)
    return GPTLerror ("%s: Error in GPTLpr_file\n", thisfunc);

  return 0;
}

/* 
** GPTLpr_file: Print values of all timers
**
** Input arguments:
**   outfile: Name of output file to write
**
** Return value: 0 (success) or GPTLerror (failure)
*/
int GPTLpr_file (const char *outfile)
{
  FILE *fp;                 // file handle to write to
  Timer *ptr;               // walk through master thread linked list
  Timer *tptr;              // walk through slave threads linked lists
  Timer sumstats;           // sum of same timer stats over threads
  Outputfmt outputfmt;      // max depth, namelen, chars2pr
  int n, t;                 // indices
  int ndup;                 // number of duplicate auto-instrumented addresses
  unsigned long totcount;   // total timer invocations
  float *sum;               // sum of overhead values (per thread)
  float osum;               // sum of overhead over threads
  bool found;               // matching name found: exit linked list traversal
  bool foundany;            // multiple threads with matching name => print across-threads results
  bool first;               // flag 1st time entry found
  double self_ohd;          // estimated library overhead in self timer
  double parent_ohd;        // estimated library overhead due to self in parent timer
  float procsiz, rss;       // returned from GPTLget_procsiz

  static const char *thisfunc = "GPTLpr_file";

  if ( ! initialized)
    return GPTLerror ("%s: GPTLinitialize() has not been called\n", thisfunc);

  // Not great hack to force output to stderr: "output" is the string "stderr"
  if (STRMATCH (outfile, "stderr") || ! (fp = fopen (outfile, "w")))
    fp = stderr;

  // Rename auto-instrumented entries with same name but different address due to lopping
  if ((ndup = GPTLrename_duplicate_addresses ()) > 0) {
    fprintf (fp, "%d duplicate auto-instrumented addresses were found and @<num> added to name",
	     ndup);
    if (ndup > 255) {
      fprintf (fp, "%d did NOT have @<num> added because there were too many\n", ndup);
      fprintf (fp, "Consider increasing the value of MAX_CHARS in private.h next run\n");
    }
  }
  // Print a warning if GPTLerror() was ever called
  if (GPTLnum_errors () > 0) {
    fprintf (fp, "WARNING: GPTLerror was called at least once during the run.\n");
    fprintf (fp, "Please examine your output for error messages beginning with GPTL...\n");
  }

  // Print a warning if imperfect nesting was encountered
  if (imperfect_nest) {
    fprintf (fp, "WARNING: SOME TIMER CALLS WERE DETECTED TO HAVE IMPERFECT NESTING.\n");
    fprintf (fp, "TIMING RESULTS WILL BE PRINTED WITHOUT INDENTING AND NO PARENT-CHILD\n");
    fprintf (fp, "INDENTING WILL BE DONE.\n");
    fprintf (fp, "ALSO: NO MULTIPLE PARENT INFORMATION WILL BE PRINTED SINCE IT MAY CONTAIN ERRORS\n");
  }

  // A set of nasty ifdefs to tell important aspects of how GPTL was built
#ifdef HAVE_NANOTIME
  if (funclist[funcidx].option == GPTLnanotime) {
    fprintf (fp, "Clock rate = %f MHz\n", cpumhz);
    fprintf (fp, "Source of clock rate was %s\n", clock_source);
    if (strcmp (clock_source, "/proc/cpuinfo") == 0) {
      fprintf (fp, "WARNING: The contents of /proc/cpuinfo can change in variable frequency CPUs");
      fprintf (fp, "Therefore the use of nanotime (register read) is not recommended on machines so equipped");
    }
#ifdef BIT64
    fprintf (fp, "  BIT64 was true\n");
#else
    fprintf (fp, "  BIT64 was false\n");
#endif
  }
#endif

#if ( defined UNDERLYING_OMP )
  fprintf (fp, "GPTL was built with UNDERLYING_OMP\n");
#elif ( defined UNDERLYING_PTHREADS )
  fprintf (fp, "GPTL was built with UNDERLYING_PTHREADS\n");
#else
  fprintf (fp, "GPTL was built without threading\n");
#endif

#ifdef HAVE_LIBMPI
  fprintf (fp, "HAVE_LIBMPI was true\n");

#ifdef ENABLE_PMPI
  fprintf (fp, "  ENABLE_PMPI was true\n");
#else
  fprintf (fp, "  ENABLE_PMPI was false\n");
#endif

#else
  fprintf (fp, "HAVE_LIBMPI was false\n");
#endif

#ifdef HAVE_PAPI
  fprintf (fp, "HAVE_PAPI was true\n");
  if (dousepapi) {
    if (GPTL_PAPIis_multiplexed ())
      fprintf (fp, "  PAPI event multiplexing was ON\n");
    else
      fprintf (fp, "  PAPI event multiplexing was OFF\n");
    GPTL_PAPIprintenabled (fp);
  }
#else
  fprintf (fp, "HAVE_PAPI was false\n");
#endif

#ifdef ENABLE_NESTEDOMP
  fprintf (fp, "ENABLE_NESTEDOMP was true\n");
#else
  fprintf (fp, "ENABLE_NESTEDOMP was false\n");
#endif

#ifdef HAVE_LIBUNWIND
  fprintf (fp, "Autoprofiling capability was enabled with libunwind\n");
#elif defined HAVE_BACKTRACE	  
  fprintf (fp, "Autoprofiling capability was enabled with backtrace\n");
#else
  fprintf (fp, "Autoprofiling capability was NOT enabled\n");
#endif	   

  fprintf (fp, "Underlying timing routine was %s.\n", funclist[funcidx].name);
  (void) GPTLget_overhead (fp, ptr2wtimefunc, getentry, genhashidx, 
			   stackidx, callstack, hashtable[0], tablesize, dousepapi, imperfect_nest, 
			   &self_ohd, &parent_ohd);
  if (dopr_preamble) {
    fprintf (fp, "\nIf overhead stats are printed, they are the columns labeled self_OH and parent_OH\n"
	     "self_OH is estimated as 2X the Fortran layer cost (start+stop) plust the cost of \n"
	     "a single call to the underlying timing routine.\n"
	     "parent_OH is the overhead for the named timer which is subsumed into its parent.\n"
	     "It is estimated as the cost of a single GPTLstart()/GPTLstop() pair.\n"
             "Print method was %s.\n", methodstr (method));
#ifdef ENABLE_PMPI
    fprintf (fp, "\nIf a AVG_MPI_BYTES field is present, it is an estimate of the per-call\n"
             "average number of bytes handled by that process.\n"
             "If timers beginning with sync_ are present, it means MPI synchronization "
             "was turned on.\n");
#endif
    fprintf (fp, "\nIf a \'%%_of\' field is present, it is w.r.t. the first timer for thread 0.\n"
             "If a \'e6_per_sec\' field is present, it is in millions of PAPI counts per sec.\n\n"
             "A '*' in column 1 below means the timer had multiple parents, though the values\n"
             "printed are for all calls. Multiple parent stats appear later in the file in the\n"
	     "section titled \'Multiple parent info\'\n"
	     "A \'!\' in column 1 means the timer is currently ON and the printed timings are only\n"
	     "valid as of the previous GPTLstop. \'!\' overrides \'*\' if the region had multiple\n"
	     "parents and was currently ON.\n\n");
  }

  // Print the process size at time of call to GPTLpr_file
  (void) GPTLget_procsiz (&procsiz, &rss);
  fprintf (fp, "Process size=%f MB rss=%f MB\n\n", procsiz, rss);

  sum = (float *) GPTLallocate (GPTLnthreads * sizeof (float), thisfunc);
  
  for (t = 0; t < GPTLnthreads; ++t) {
    print_titles (t, fp, &outputfmt);
    /*
    ** Print timing stats. If imperfect nesting was detected, print stats by going through
    ** the linked list and do not indent anything due to the possibility of error.
    ** Otherwise, print call tree and properly indented stats via recursive routine. "-1" 
    ** is flag to avoid printing dummy outermost timer, and initialize the depth.
    */
    if (imperfect_nest) {
      for (ptr = timers[t]->next; ptr; ptr = ptr->next) {
	printstats (ptr, fp, t, 0, false, self_ohd, parent_ohd, outputfmt);
      }
    } else {
      printself_andchildren (timers[t], fp, t, -1, self_ohd, parent_ohd, outputfmt);
    }

    // Sum of self+parent overhead across timers is an estimate of total overhead.
    sum[t]   = 0;
    totcount = 0;
    for (ptr = timers[t]->next; ptr; ptr = ptr->next) {
      sum[t]   += ptr->count * (parent_ohd + self_ohd);
      totcount += ptr->count;
    }
    if (wallstats.enabled && overheadstats.enabled)
      fprintf (fp, "\n");
    fprintf (fp, "Overhead sum = %9.3g wallclock seconds\n", sum[t]);
    if (totcount < PRTHRESH)
      fprintf (fp, "Total calls  = %lu\n", totcount);
    else
      fprintf (fp, "Total calls  = %9.3e\n", (float) totcount);
  }

  // Print per-name stats for all threads
  if (dopr_threadsort && GPTLnthreads > 1) {
    int nblankchars;
    fprintf (fp, "\nSame stats sorted by timer for threaded regions:\n");
    fprintf (fp, "Thd ");

    // Reset outputfmt contents for multi-threads
    outputfmt.max_depth    = 0;
    outputfmt.max_namelen  = get_longest_omp_namelen ();
    outputfmt.max_chars2pr = outputfmt.max_namelen;
    nblankchars            = outputfmt.max_chars2pr + 1; // + 1 ensures a blank after name
    for (n = 0; n < nblankchars; ++n)  // length of longest multithreaded timer name
      fprintf (fp, " ");

    fprintf (fp, "  Called  Recurse");

    if (cpustats.enabled)
      fprintf (fp, "%9s", cpustats.str);
    if (wallstats.enabled) {
      fprintf (fp, "%9s", wallstats.str);
      if (percent && timers[0]->next)
        fprintf (fp, " %%_of_%4.4s ", timers[0]->next->name);
      if (overheadstats.enabled)
        fprintf (fp, "%9s", overheadstats.str);
    }

#ifdef HAVE_PAPI
    GPTL_PAPIprstr (fp);
#endif

#ifdef COLLIDE
    fprintf (fp, "  Collide");
#endif

    fprintf (fp, "\n");
    // Start at next to skip GPTL_ROOT
    for (ptr = timers[0]->next; ptr; ptr = ptr->next) {      
      // To print sum stats, first create a new timer then copy thread 0
      // stats into it. then sum using "add", and finally print.
      foundany = false;
      first = true;
      sumstats = *ptr;
      for (t = 1; t < GPTLnthreads; ++t) {
        found = false;
        for (tptr = timers[t]->next; tptr && ! found; tptr = tptr->next) {
          if (STRMATCH (ptr->name, tptr->name)) {
            // Only print thread 0 when this timer found for other threads
            if (first) {
              first = false;
              fprintf (fp, "%3.3d ", 0);
              printstats (ptr, fp, 0, 0, false, self_ohd, parent_ohd, outputfmt);
            }

            found = true;
            foundany = true;
            fprintf (fp, "%3.3d ", t);
            printstats (tptr, fp, 0, 0, false, self_ohd, parent_ohd, outputfmt);
            add (&sumstats, tptr);
          }
        }
      }

      if (foundany) {
        fprintf (fp, "SUM ");
        printstats (&sumstats, fp, 0, 0, false, self_ohd, parent_ohd, outputfmt);
        fprintf (fp, "\n");
      }
    }

    // Repeat overhead print in loop over threads
    if (wallstats.enabled && overheadstats.enabled) {
      osum = 0.;
      for (t = 0; t < GPTLnthreads; ++t) {
        fprintf (fp, "OVERHEAD.%3.3d (wallclock seconds) = %9.3g\n", t, sum[t]);
        osum += sum[t];
      }
      fprintf (fp, "OVERHEAD.SUM (wallclock seconds) = %9.3g\n", osum);
    }
  }

  // For auto-instrumented apps, translate names which have been truncated for output formatting
  for (t = 0; t < GPTLnthreads; ++t)
    translate_truncated_names (t, fp);
  
  // Print info about timers with multiple parents ONLY if imperfect nesting was not discovered
  if (dopr_multparent && ! imperfect_nest) {
    for (t = 0; t < GPTLnthreads; ++t) {
      bool some_multparents = false;   // thread has entries with multiple parents?
      for (ptr = timers[t]->next; ptr; ptr = ptr->next) {
        if (ptr->nparent > 1) {
          some_multparents = true;
          break;
        }
      }

      if (some_multparents) {
        fprintf (fp, "\nMultiple parent info for thread %d:\n", t);
        if (dopr_preamble && t == 0) {
          fprintf (fp, "Columns are count and name for the listed child\n"
                   "Rows are each parent, with their common child being the last entry, "
                   "which is indented.\n"
                   "Count next to each parent is the number of times it called the child.\n"
                   "Count next to child is total number of times it was called by the "
                   "listed parents.\n\n");
        }

        for (ptr = timers[t]->next; ptr; ptr = ptr->next)
          if (ptr->nparent > 1)
            print_multparentinfo (fp, ptr);
      }
    }
  }

  // Print hash table stats
  if (dopr_collision)
    GPTLprint_hashstats (fp, GPTLnthreads, hashtable, tablesize);

  // Print stats on GPTL memory usage
  GPTLprint_memstats (fp, timers, tablesize);

  free (sum);

  if (fp != stderr && fclose (fp) != 0)
    fprintf (stderr, "%s: Attempt to close %s failed\n", thisfunc, outfile);

  return 0;
}

/* 
** GPTLrename_duplcicate_addresses: Create unique name for auto-profiled entries that are 
**                                  now duplicates due to truncation
**
** Return value: max number of duplicates found
*/
int GPTLrename_duplicate_addresses ()
{
  Timer *ptr;       // iterate through timers
  Timer *testptr;   // start at ptr, iterate through remaining timers
  int nfound = 0;   // number of duplicates found
  int t;            // thread index
  static const char *thisfunc = "GPTLrename_duplicate_addresses";

  for (t = 0; t < GPTLnthreads; ++t) {
    for (ptr = timers[t]; ptr; ptr = ptr->next) {
      unsigned int idx = 0;
      // Check only entries with a longname and don't have '@' in their name
      // The former means auto-instrumented
      // The latter means the name hasn't yet had the proper '@<number>' appended.
      if (ptr->longname && ! index (ptr->name, '@')) {
	for (testptr = ptr->next; testptr; testptr = testptr->next) {
	  if (testptr->longname && STRMATCH (ptr->name, testptr->name)) {
	    // Add string "@<idx>" to end of testptr->name to indicate duplicate auto-profiled
	    // name for multiple addresses. Probable inlining issue.
	    if (++idx < 4096)    // 4095 is the max integer printable in 3 hex chars
	      snprintf (&testptr->name[MAX_CHARS-4], 5, "@%X", idx);
	    else
	      snprintf (&testptr->name[MAX_CHARS-4], 5, "@MAX");
	  }
	}
	// @<number> has been added to duplicates. Now add @0 to original
	snprintf (&ptr->name[MAX_CHARS-4], 5, "@%X", 0);
	nfound = MAX (nfound, idx);
      }
    }
  }
  return nfound;
}

// translate_truncated_names: For auto-profiled entries, print to output file the translation of
//                            truncated names to full signature
static void translate_truncated_names (int t, FILE *fp)
{
  Timer *ptr;

  fprintf (fp, "thread %d long name translations (empty when no auto-instrumentation):\n", t);
  for (ptr = timers[t]->next; ptr; ptr = ptr->next) {
    if (ptr->longname)
      fprintf (fp, "%s = %s\n", ptr->name, ptr->longname);
  }
}

// get_longest_omp_namelen: Discover longest name shared across threads
static int get_longest_omp_namelen (void)
{
  Timer *ptr;
  Timer *tptr;
  bool found;
  int t;
  int longest = 0;

  for (ptr = timers[0]->next; ptr; ptr = ptr->next) {
    for (t = 1; t < GPTLnthreads; ++t) {
      found = false;
      for (tptr = timers[t]->next; tptr && ! found; tptr = tptr->next) {
	if (STRMATCH (tptr->name, ptr->name)) {
	  found = true;
	  longest = MAX (longest, strlen (ptr->name));
	}
	if (found) // Found matching name: done with this thread
	  break;
      }
      if (found)   // No need to check other threads for the same name
	break;
    }
  }
  return longest;
}

/* 
** print_titles: Print headings to output file. If imperfect nesting was detected, print simply by
**               following the linked list. Otherwise, indent using parent-child relationships.
**
** Input arguments:
**   t:         thread number
**   fp:        file pointer to write to
**   outputfmt: max depth, namelen, chars2pr
*/
static void print_titles (int t, FILE *fp, Outputfmt *outputfmt)
{
  int n;
  int ret;
  int nblankchars;
  static const char *thisfunc = "print_titles";

  /*
  ** Construct tree for printing timers in parent/child form. get_outputfmt() must be called 
  ** AFTER construct_tree() because it relies on the per-parent children arrays being complete.
  ** Initialize outputfmt, then call recursive routine to fill its contents
  */
  outputfmt->max_depth    = 0;
  outputfmt->max_namelen  = 0;
  outputfmt->max_chars2pr = 0;

  if (imperfect_nest) {
    outputfmt->max_namelen  = get_max_namelen (timers[t]);
    outputfmt->max_chars2pr = outputfmt->max_namelen;
    nblankchars             = outputfmt->max_namelen + 1;
  } else {
    if (construct_tree (timers[t], method) != 0)
      printf ("GPTL: %s: failure from construct_tree: output will be incomplete\n", thisfunc);

    // Start at GPTL_ROOT because that is the parent of all timers => guarantee traverse
    // full tree. -1 is initial call tree depth
    ret = get_outputfmt (timers[t], -1, indent_chars, outputfmt);  // -1 is initial call tree depth
#ifdef DEBUG
    printf ("%s t=%d got outputfmt=%d %d %d\n",
	    thisfunc, t, outputfmt->max_depth, outputfmt->max_namelen, outputfmt->max_chars2pr);
#endif
    nblankchars = indent_chars + outputfmt->max_chars2pr + 1;
  }

  if (t > 0)
    fprintf (fp, "\n");
  fprintf (fp, "Stats for thread %d:\n", t);

  // Start title printing after longest (indent_chars + name) + 1
  for (n = 0; n < nblankchars; ++n)
    fprintf (fp, " ");
  fprintf (fp, "  Called  Recurse");

  // Print strings for enabled timer types
  if (cpustats.enabled)
    fprintf (fp, "%9s", cpustats.str);
  if (wallstats.enabled) {
    fprintf (fp, "%9s", wallstats.str);
    if (percent && timers[0]->next)
      fprintf (fp, " %%_of_%4.4s", timers[0]->next->name);
    if (overheadstats.enabled)
      fprintf (fp, "%9s", overheadstats.str);
  }
#ifdef ENABLE_PMPI
  fprintf (fp, " AVG_MPI_BYTES");
#endif

#ifdef HAVE_PAPI
  GPTL_PAPIprstr (fp);
#endif

#ifdef COLLIDE
    fprintf (fp, "  Collide");
#endif

  fprintf (fp, "\n");
  return;
}

/* 
** construct_tree: Build the parent->children tree starting with knowledge of
**                 parent list for each child.
**
** Input arguments:
**   method:  method to be used to define the links
**
** Input/Output arguments:
**   timerst: Linked list of timers. "children" array for each timer will be constructed
**
** Return value: 0 (success) or GPTLerror (failure)
*/
int construct_tree (Timer *timerst, GPTLMethod method)
{
  Timer *ptr;
  Timer *pptr = 0;  // parent pointe (init to NULL to avoid compiler warning)
  int nparent;      // number of parents
  int maxcount;     // max calls by a single parent
  int n;            // loop over nparent
  static const char *thisfunc = "construct_tree";

  // Walk the linked list to build the parent-child tree, using whichever
  // mechanism is in place. newchild() will prevent loops.
  for (ptr = timerst; ptr; ptr = ptr->next) {
    switch (method) {
    case GPTLfirst_parent:
      if (ptr->nparent > 0) {
        pptr = ptr->parent[0];
        if (newchild (pptr, ptr) != 0) {};
      }
      break;
    case GPTLlast_parent:
      if (ptr->nparent > 0) {
        nparent = ptr->nparent;
        pptr = ptr->parent[nparent-1];
        if (newchild (pptr, ptr) != 0) {};
      }
      break;
    case GPTLmost_frequent:
      maxcount = 0;
      for (n = 0; n < ptr->nparent; ++n) {
        if (ptr->parent_count[n] > maxcount) {
          pptr = ptr->parent[n];
          maxcount = ptr->parent_count[n];
        }
      }
      if (maxcount > 0) {   // not an orphan
        if (newchild (pptr, ptr) != 0) {};
      }
      break;
    case GPTLfull_tree:
      for (n = 0; n < ptr->nparent; ++n) {
        pptr = ptr->parent[n];
        if (newchild (pptr, ptr) != 0) {};
      }
      break;
    default:
      return GPTLerror ("GPTL: %s: method %d is not known\n", thisfunc, method);
    }
  }
  return 0;
}

// methodstr: Return a pointer to a string which represents the print method
static char *methodstr (GPTLMethod method)
{
  static char first_parent[]  = "first_parent";
  static char last_parent[]   = "last_parent";
  static char most_frequent[] = "most_frequent";
  static char full_tree[]     = "full_tree";

  if (method == GPTLfirst_parent)
    return first_parent;
  else if (method == GPTLlast_parent)
    return last_parent;
  else if (method == GPTLmost_frequent)
    return most_frequent;
  else if (method == GPTLfull_tree)
    return full_tree;
  else
    return unknown;
}

/* 
** newchild: Add an entry to the children list of parent. Use function
**   is_descendant() to prevent infinite loops. 
**
** Input arguments:
**   child:  child to be added
**
** Input/output arguments:
**   parent: parent node which will have "child" added to its "children" array
**
** Return value: 0 (success) or GPTLerror (failure)
*/
static int newchild (Timer *parent, Timer *child)
{
  int nchildren;     // number of children (temporary)
  Timer **chptr;     // array of pointers to children
  static const char *thisfunc = "newchild";

  if (parent == child)
    return GPTLerror ("%s: child %s can't be a parent of itself\n", thisfunc, child->name);

  // To guarantee no loops, ensure that proposed parent isn't already a descendant of 
  // proposed child
  if (is_descendant (child, parent)) {
    return GPTLerror ("GPTL: %s: loop detected: NOT adding %s to descendant list of %s. "
                      "Proposed parent is in child's descendant path.\n",
                      thisfunc, child->name, parent->name);
  }

  // Add child to parent's array of children if it isn't already there (e.g. by an earlier call
  // to GPTLpr*)
  if ( ! is_onlist (child, parent)) {
    ++parent->nchildren;
    nchildren = parent->nchildren;
    chptr = (Timer **) realloc (parent->children, nchildren * sizeof (Timer *));
    if ( ! chptr)
      return GPTLerror ("%s: realloc error\n", thisfunc);
    parent->children = chptr;
    parent->children[nchildren-1] = child;
  }

  return 0;
}

/* 
** get_outputfmt: Determine max depth, name length, and chars to be printed prior to data by 
**   traversing tree recursively
**
** Input arguments:
**   ptr:   Starting timer
**   depth: current depth when function invoked 
**
** Return value: maximum characters to be printed prior to data
*/
static int get_outputfmt (const Timer *ptr, const int depth, const int indent, 
			  Outputfmt *outputfmt)
{
  int ret;
  int namelen  = strlen (ptr->name);
  int chars2pr = namelen + indent*depth;
  int n;

  if (ptr->nchildren == 0) {
    fill_output (depth, namelen, chars2pr, outputfmt);
    return 0;
  }

  for (n = 0; n < ptr->nchildren; ++n) {
    ret = get_outputfmt (ptr->children[n], depth+1, indent, outputfmt);
    fill_output (depth, namelen, chars2pr, outputfmt);
  }
  return 0;
}

static void fill_output (int depth, int namelen, int chars2pr, Outputfmt *outputfmt)
{
  if (depth > outputfmt->max_depth)
    outputfmt->max_depth = depth;

  if (namelen > outputfmt->max_namelen)
    outputfmt->max_namelen = namelen;

  if (chars2pr > outputfmt->max_chars2pr)
    outputfmt->max_chars2pr = chars2pr;
}

/*
** get_max_namelen: Discover maximum name length. Called only when imperfect_nest is true
** Input arguments:
**   ptr: Starting timer
**
** Return value:
**   max name length
*/
static int get_max_namelen (Timer *timers)
{
  Timer *ptr;            // traverse linked list
  int namelen;           // name length for individual timer
  int max_namelen = 0;   // return value

  for (ptr = timers; ptr; ptr = ptr->next) {
    namelen = strlen (ptr->name);
    if (namelen > max_namelen)
      max_namelen = namelen;
  }
  return max_namelen;
}

/* 
** is_descendant: Determine whether node2 is in the descendant list for node1
**
** Input arguments:
**   node1: starting node for recursive search
**   node2: node to be searched for
**
** Return value: true or false
*/
static int is_descendant (const Timer *node1, const Timer *node2)
{
  int n;

  // Breadth before depth for efficiency
  for (n = 0; n < node1->nchildren; ++n)
    if (node1->children[n] == node2)
      return 1;

  for (n = 0; n < node1->nchildren; ++n)
    if (is_descendant (node1->children[n], node2))
      return 1;

  return 0;
}

/* 
** is_onlist: Determine whether child is in parent's list of children
**
** Input arguments:
**   child: who to search for
**   parent: search through his list of children
**
** Return value: true or false
*/
static int is_onlist (const Timer *child, const Timer *parent)
{
  int n;

  for (n = 0; n < parent->nchildren; ++n) {
    if (child == parent->children[n])
      return 1;
  }
  return 0;
}

/* 
** printstats: print a single timer
**
** Input arguments:
**   timer:        timer for which to print stats
**   fp:           file descriptor to write to
**   t:            thread number
**   depth:        depth to indent timer
**   doindent:     whether indenting will be done
**   tot_overhead: underlying timing routine overhead
*/
static void printstats (const Timer *timer, FILE *fp, int t, int depth, bool doindent,
                        double self_ohd, double parent_ohd, const Outputfmt outputfmt)
{
  int i;               // index
  int indent;          // index for indenting
  int extraspace;      // for padding to length of longest name
  float fusr;          // user time as float
  float fsys;          // system time as float
  float usrsys;        // usr + sys
  float elapse;        // elapsed time
  float wallmax;       // max wall time
  float wallmin;       // min wall time
  float ratio;         // percentage calc
  static const char *thisfunc = "printstats";

  if (timer->onflg && verbose)
    fprintf (stderr, "GPTL: %s: timer %s had not been turned off\n", thisfunc, timer->name);

  // Flag regions having multiple parents with a "*" in column 1
  // TODO: The following ASSUMES indent_chars=2!!!
  if (doindent) {
    if (timer->onflg)
      fprintf (fp, "! ");
    else if (timer->nparent > 1)
      fprintf (fp, "* ");
    else
      fprintf (fp, "  ");

    // Indent to depth of this timer
    for (indent = 0; indent < depth*indent_chars; ++indent)
      fprintf (fp, " ");
  }

  fprintf (fp, "%s", timer->name);

  // Pad to most chars to print
  extraspace = outputfmt.max_chars2pr - (depth*indent_chars + strlen (timer->name));
  for (i = 0; i < extraspace; ++i)
    fprintf (fp, " ");

  if (timer->count < PRTHRESH) {
    if (timer->nrecurse > 0)
      fprintf (fp, " %8lu %8lu", timer->count, timer->nrecurse);
    else
      fprintf (fp, " %8lu     -   ", timer->count);
  } else {
    if (timer->nrecurse > 0)
      fprintf (fp, " %8.1e %8.0e",    (float) timer->count, (float) timer->nrecurse);
    else
      fprintf (fp, " %8.1e     -   ", (float) timer->count);
  }

  if (cpustats.enabled) {
    fusr = timer->cpu.accum_utime / (float) ticks_per_sec;
    fsys = timer->cpu.accum_stime / (float) ticks_per_sec;
    usrsys = fusr + fsys;
    fprintf (fp, " %8.1e %8.1e %8.1e", fusr, fsys, usrsys);
  }

  if (wallstats.enabled) {
    elapse = timer->wall.accum;
    wallmax = timer->wall.max;
    wallmin = timer->wall.min;

    if (elapse < 0.01)
      fprintf (fp, " %8.2e", elapse);
    else
      fprintf (fp, " %8.3f", elapse);

    if (wallmax < 0.01)
      fprintf (fp, " %8.2e", wallmax);
    else
      fprintf (fp, " %8.3f", wallmax);

    if (wallmin < 0.01)
      fprintf (fp, " %8.2e", wallmin);
    else
      fprintf (fp, " %8.3f", wallmin);

    if (percent && timers[0]->next) {
      ratio = 0.;
      if (timers[0]->next->wall.accum > 0.)
        ratio = (timer->wall.accum * 100.) / timers[0]->next->wall.accum;
      fprintf (fp, " %8.2f", ratio);
    }

    if (overheadstats.enabled) {
      fprintf (fp, " %8.3f %8.3f", timer->count*self_ohd, timer->count*parent_ohd);
    }
  }

#ifdef ENABLE_PMPI
  if (timer->nbytes == 0.)
    fprintf (fp, "       -      ");
  else
    fprintf (fp, "%13.3e ", timer->nbytes / timer->count);
#endif
  
#ifdef HAVE_PAPI
  GPTL_PAPIpr (fp, &timer->aux, t, timer->count, timer->wall.accum);
#endif

#ifdef COLLIDE
  if (timer->collide > PRTHRESH)
    fprintf (fp, " %8.1e", (float) timer->collide);
  else if (timer->collide > 0)
    fprintf (fp, " %8lu", timer->collide);
#endif
  
  fprintf (fp, "\n");
}

// print_multparentinfo: print info about regions with multiple parents
void print_multparentinfo (FILE *fp, Timer *ptr)
{
  int n;

  if (ptr->norphan > 0) {
    if (ptr->norphan < PRTHRESH)
      fprintf (fp, "%8u %-32s\n", ptr->norphan, "ORPHAN");
    else
      fprintf (fp, "%8.1e %-32s\n", (float) ptr->norphan, "ORPHAN");
  }

  for (n = 0; n < ptr->nparent; ++n) {
    char *parentname;
    if (ptr->parent[n]->longname)
      parentname = ptr->parent[n]->longname;
    else
      parentname = ptr->parent[n]->name;
    
    if (ptr->parent_count[n] < PRTHRESH)
      fprintf (fp, "%8d %-s\n", ptr->parent_count[n], parentname);
    else
      fprintf (fp, "%8.1e %-s\n", (float) ptr->parent_count[n], parentname);
  }

  if (ptr->count < PRTHRESH)
    if (ptr->longname)
      fprintf (fp, "%8lu   %-s\n\n", ptr->count, ptr->longname);
    else
      fprintf (fp, "%8lu   %-s\n\n", ptr->count, ptr->name);
  else
    if (ptr->longname)
      fprintf (fp, "%8.1e   %-s\n\n", (float) ptr->count, ptr->longname);
    else
      fprintf (fp, "%8.1e   %-s\n\n", (float) ptr->count, ptr->name);
}

// add: add the contents of timer tin to timer tout
static void add (Timer *tout, const Timer *tin)
{
  tout->count += tin->count;

  if (wallstats.enabled) {
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
  return GPTLerror ("GPTL: get_cpustamp: times() not available\n");
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
**   t:           thread number (if < 0, the request is for the current thread)
**
** Output args:
**   count:            number of times this timer was called
**   onflg:            whether timer is currently on
**   wallclock:        accumulated wallclock time
**   usr:              accumulated user CPU time
**   sys:              accumulated system CPU time
**   papicounters_out: accumulated PAPI counters
*/
int GPTLquery (const char *name, int t, int *count, int *onflg, double *wallclock,
               double *dusr, double *dsys, long long *papicounters_out, const int maxcounters)
{
  Timer *ptr;
  unsigned int indx;
  static const char *thisfunc = "GPTLquery";
  
  if ( ! initialized)
    return GPTLerror ("%s: GPTLinitialize has not been called\n", thisfunc);
  
  // If t is < 0, assume the request is for the current thread
  if (t < 0) {
    if ((t = GPTLget_thread_num ()) < 0)
      return GPTLerror ("%s: GPTLget_thread_num failure\n", thisfunc);
  } else {
    if (t >= GPTLmax_threads)
      return GPTLerror ("%s: requested thread %d is too big\n", thisfunc, t);
  }

  indx = genhashidx (name);
  ptr = getentry (hashtable[t], name, indx);
  if ( !ptr)
    return GPTLerror ("%s: requested timer %s does not have a name hash\n", thisfunc, name);

  *onflg     = ptr->onflg;
  *count     = ptr->count;
  *wallclock = ptr->wall.accum;
  *dusr      = ptr->cpu.accum_utime / (double) ticks_per_sec;
  *dsys      = ptr->cpu.accum_stime / (double) ticks_per_sec;
#ifdef HAVE_PAPI
  GPTL_PAPIquery (&ptr->aux, papicounters_out, maxcounters);
#endif
  return 0;
}

/*
** GPTLget_wallclock: return wallclock accumulation for a timer.
** 
** Input args:
**   timername: timer name
**   t:         thread number (if < 0, the request is for the current thread)
**
** Output args:
**   value: current wallclock accumulation for the timer
*/
int GPTLget_wallclock (const char *timername, int t, double *value)
{
  Timer *ptr;
  unsigned int indx;   // hash index returned from getentry (unused)
  static const char *thisfunc = "GPTLget_wallclock";
  
  if ( ! initialized)
    return GPTLerror ("%s: GPTLinitialize has not been called\n", thisfunc);

  if ( ! wallstats.enabled)
    return GPTLerror ("%s: wallstats not enabled\n", thisfunc);
  
  // If t is < 0, assume the request is for the current thread
  if (t < 0) {
    if ((t = GPTLget_thread_num ()) < 0)
      return GPTLerror ("%s: bad return from GPTLget_thread_num\n", thisfunc);
  } else {
    if (t >= GPTLmax_threads)
      return GPTLerror ("%s: requested thread %d is too big\n", thisfunc, t);
  }
  
  indx = genhashidx (timername);
  ptr = getentry (hashtable[t], timername, indx);
  if ( ! ptr)
    return GPTLerror ("%s: requested timer %s does not exist (or auto-instrumented?)\n",
		      thisfunc, timername);
  *value = ptr->wall.accum;
  return 0;
}

/*
** GPTLget_wallclock_latest: return most recent wallclock value for a timer.
** 
** Input args:
**   timername: timer name
**   t:         thread number (if < 0, the request is for the current thread)
**
** Output args:
**   value: most recent wallclock value for the timer
*/
int GPTLget_wallclock_latest (const char *timername, int t, double *value)
{
  Timer *ptr;
  unsigned int indx;   // hash index returned from getentry (unused)
  static const char *thisfunc = "GPTLget_wallclock_latest";
  
  if ( ! initialized)
    return GPTLerror ("%s: GPTLinitialize has not been called\n", thisfunc);

  if ( ! wallstats.enabled)
    return GPTLerror ("%s: wallstats not enabled\n", thisfunc);
  
  // If t is < 0, assume the request is for the current thread
  if (t < 0) {
    if ((t = GPTLget_thread_num ()) < 0)
      return GPTLerror ("%s: bad return from GPTLget_thread_num\n", thisfunc);
  } else {
    if (t >= GPTLmax_threads)
      return GPTLerror ("%s: requested thread %d is too big\n", thisfunc, t);
  }
  
  indx = genhashidx (timername);
  ptr = getentry (hashtable[t], timername, indx);
  if ( !ptr)
    return GPTLerror ("%s: requested timer %s does not exist\n", thisfunc, timername);
  *value = ptr->wall.latest;
  return 0;
}

/*
** GPTLget_threadwork: For a timer, across threads compute max work and imbalance
**
** Input arguments:
**   name: timer name
**
** Output arguments:
**   maxwork: maximum work across threads
**   imbal:   imbalance vs. perfectly distributed workload
**
** Return value: 0 (success) or -1 (failure)
*/
int GPTLget_threadwork (const char *name, double *maxwork, double *imbal)
{
  Timer *ptr;                  // linked list pointer
  int t;                       // thread number for this process
  int nfound = 0;              // number of threads which did work (must be > 0
  unsigned int indx;           // index into hash table
  double innermax = 0.;        // maximum work across threads
  double totalwork = 0.;       // total work done by all threads
  double balancedwork;         // time if work were perfectly load balanced
  static const char *thisfunc = "GPTLget_threadwork";

  if (disabled)
    return 0;

  if ( ! initialized)
    return GPTLerror ("%s: GPTLinitialize has not been called\n", thisfunc);

  if ( ! wallstats.enabled)
    return GPTLerror ("%s: wallstats must be enabled to call this function\n", thisfunc);

  if (GPTLget_thread_num () != 0)
    return GPTLerror ("%s: Must be called by the master thread\n", thisfunc);

  indx = genhashidx (name);
  for (t = 0; t < GPTLnthreads; ++t) {
    ptr = getentry (hashtable[t], name, indx);
    if (ptr) {
      ++nfound;
      innermax = MAX (innermax, ptr->wall.accum);
      totalwork += ptr->wall.accum;
    }
  }

  // It's an error to call this routine for a region that does not exist
  if (nfound == 0)
    return GPTLerror ("%s: No entries exist for name=%s\n", thisfunc, name);

  // A perfectly load-balanced calculation would take time=totalwork/GPTLnthreads
  // Therefore imbalance is slowest thread time minus this number
  balancedwork = totalwork / GPTLnthreads;
  *maxwork = innermax;
  *imbal = innermax - balancedwork;

  return 0;
}

/*
** GPTLstartstop_val: Take user input to treat as the result of calling start/stop
**
** Input arguments:
**   name: timer name
**   value: value to add to the timer
**
** Return value: 0 (success) or -1 (failure)
*/
int GPTLstartstop_val (const char *name, double value)
{
  Timer *ptr;
  int t;
  unsigned int indx;         // index into hash table
  static const char *thisfunc = "GPTLstartstop_val";

  if (disabled)
    return 0;

  if ( ! initialized)
    return GPTLerror ("%s: GPTLinitialize has not been called\n", thisfunc);

  if ( ! wallstats.enabled)
    return GPTLerror ("%s: wallstats must be enabled to call this function\n", thisfunc);

  if (value < 0.)
    return GPTLerror ("%s: Input value must not be negative\n", thisfunc);

  if ((t = GPTLget_thread_num ()) < 0)
    return GPTLerror ("%s: bad return from GPTLget_thread_num\n", thisfunc);

  // Find out if the timer already exists
  indx = genhashidx (name);
  ptr = getentry (hashtable[t], name, indx);

  if (ptr) {
    // The timer already exists. Bump the count manually, update the time stamp,
    // and let control jump to the point where wallclock settings are adjusted.
    ++ptr->count;
    ptr->wall.last = (*ptr2wtimefunc) ();
  } else {
    // Need to call start/stop to set up linked list and hash table.
    // "count" and "last" will also be set properly by the call to this pair.
    if (GPTLstart (name) != 0)
      return GPTLerror ("%s: Error from GPTLstart\n", thisfunc);

    if (GPTLstop (name) != 0)
      return GPTLerror ("%s: Error from GPTLstop\n", thisfunc);

    // start/stop pair just called should guarantee ptr will be found
    if ( ! (ptr = getentry (hashtable[t], name, indx)))
      return GPTLerror ("%s: Unexpected error from getentry\n", thisfunc);

    ptr->wall.min = value; // Since this is the first call, set min to user input
    // Minor mod: Subtract the overhead of the above start/stop call, before
    // adding user input
    ptr->wall.accum -= ptr->wall.latest;
  }

  // Overwrite the values with user input
  ptr->wall.accum += value;
  ptr->wall.latest = value;
  if (value > ptr->wall.max)
    ptr->wall.max = value;

  // On first call this setting is unnecessary but avoid an "if" test for efficiency
  if (value < ptr->wall.min)
    ptr->wall.min = value;

  return 0;
}

/*
** GPTLget_count: return number of start/stop calls for a timer.
** 
** Input args:
**   timername: timer name
**   t:         thread number (if < 0, the request is for the current thread)
**
** Output args:
**   count: current number of start/stop calls for the timer
*/
int GPTLget_count (const char *timername, int t, int *count)
{
  Timer *ptr;
  unsigned int indx;   // hash index returned from getentry (unused)
  static const char *thisfunc = "GPTLget_count";
  
  if ( ! initialized)
    return GPTLerror ("%s: GPTLinitialize has not been called\n", thisfunc);

  // If t is < 0, assume the request is for the current thread
  if (t < 0) {
    if ((t = GPTLget_thread_num ()) < 0)
      return GPTLerror ("%s: bad return from GPTLget_thread_num\n", thisfunc);
  } else {
    if (t >= GPTLmax_threads)
      return GPTLerror ("%s: requested thread %d is too big\n", thisfunc, t);
  }
  
  indx = genhashidx (timername);
  ptr = getentry (hashtable[t], timername, indx);
  if ( ! ptr)
    return GPTLerror ("%s: requested timer %s does not exist (or auto-instrumented?)\n",
		      thisfunc, timername);
  *count = ptr->count;
  return 0;
}

/*
** GPTLget_eventvalue: return PAPI-based event value for a timer. All values will be
**   returned as doubles, even if the event is not derived.
** 
** Input args:
**   timername: timer name
**   eventname: event name (must be currently enabled)
**   t:         thread number (if < 0, the request is for the current thread)
**
** Output args:
**   value: current value of the event for this timer
*/
int GPTLget_eventvalue (const char *timername, const char *eventname, int t, double *value)
{
  Timer *ptr;
  unsigned int indx;   // hash index returned from getentry (unused)
  static const char *thisfunc = "GPTLget_eventvalue";
  
  if ( ! initialized)
    return GPTLerror ("%s: GPTLinitialize has not been called\n", thisfunc);
  
  // If t is < 0, assume the request is for the current thread
  if (t < 0) {
    if ((t = GPTLget_thread_num ()) < 0)
      return GPTLerror ("%s: GPTLget_thread_num failure\n", thisfunc);
  } else {
    if (t >= GPTLmax_threads)
      return GPTLerror ("%s: requested thread %d is too big\n", thisfunc, t);
  }
  
  indx = genhashidx (timername);
  ptr = getentry (hashtable[t], timername, indx);
  if ( ! ptr)
    return GPTLerror ("%s: requested timer %s does not exist (or auto-instrumented?)\n",
		      thisfunc, timername);

#ifdef HAVE_PAPI
  return GPTL_PAPIget_eventvalue (eventname, &ptr->aux, value);
#else
  return GPTLerror ("%s: PAPI not enabled\n", thisfunc); 
#endif
}

/*
** GPTLget_nregions: return number of regions (i.e. timer names) for this thread
** 
** Input args:
**   t:    thread number (if < 0, the request is for the current thread)
**
** Output args:
**   nregions: number of regions
*/
int GPTLget_nregions (int t, int *nregions)
{
  Timer *ptr;
  static const char *thisfunc = "GPTLget_nregions";

  if ( ! initialized)
    return GPTLerror ("%s: GPTLinitialize has not been called\n", thisfunc);
  
  // If t is < 0, assume the request is for the current thread
  if (t < 0) {
    if ((t = GPTLget_thread_num ()) < 0)
      return GPTLerror ("%s: GPTLget_thread_num failure\n", thisfunc);
  } else {
    if (t >= GPTLmax_threads)
      return GPTLerror ("%s: requested thread %d is too big\n", thisfunc, t);
  }
  
  *nregions = 0;
  for (ptr = timers[t]->next; ptr; ptr = ptr->next) 
    ++*nregions;

  return 0;
}

/*
** GPTLget_regionname: return region name for this thread
** 
** Input args:
**   t:      thread number (if < 0, the request is for the current thread)
**   region: region number
**   nc:     max number of chars to put in name
**
** Output args:
**   name    region name
*/
int GPTLget_regionname (int t, int region, char *name, int nc)
{
  int ncpy;    // number of characters to copy
  int i;
  Timer *ptr;
  static const char *thisfunc = "GPTLget_regionname";

  if ( ! initialized)
    return GPTLerror ("%s: GPTLinitialize has not been called\n", thisfunc);
  
  // If t is < 0, assume the request is for the current thread
  if (t < 0) {
    if ((t = GPTLget_thread_num ()) < 0)
      return GPTLerror ("%s: GPTLget_thread_num failure\n", thisfunc);
  } else {
    if (t >= GPTLmax_threads)
      return GPTLerror ("%s: requested thread %d is too big\n", thisfunc, t);
  }
  
  ptr = timers[t]->next;
  for (i = 0; i < region; i++) {
    if ( ! ptr)
      return GPTLerror ("%s: timer number %d does not exist in thread %d\n", thisfunc, region, t);
    ptr = ptr->next;
  }

  if (ptr) {
    ncpy = MIN (nc, strlen (ptr->name));
    strncpy (name, ptr->name, ncpy);
    
    // Adding the \0 is only important when called from C
    if (ncpy < nc)
      name[ncpy] = '\0';
  } else {
    return GPTLerror ("%s: timer number %d does not exist in thread %d\n", thisfunc, region, t);
  }
  return 0;
}

// Whether GPTL has been initialized
int GPTLis_initialized (void) {return (int) initialized;}

/*
** getentry_instr: find hash table entry and return a pointer to it
**
** Input args:
**   hashtable: the hashtable (array)
**   self:      input address (from -finstrument-functions)
** Output args:
**   indx:      hashtable index
**
** Return value: pointer to the entry, or NULL if not found
*/
static inline Timer *getentry_instr (const Hashentry *hashtable, void *self, unsigned int *indx)
{
  int i;
  Timer *ptr = NULL;  // init to return value when entry not found

  /*
  ** Hash index is timer address modulo the table size
  ** On most machines, right-shifting the address helps because linkers often
  ** align functions on even boundaries
  */
  *indx = (((unsigned long) self) >> 4) % tablesize;
  for (i = 0; i < hashtable[*indx].nument; ++i) {
    if (hashtable[*indx].entries[i]->address == self) {
      ptr = hashtable[*indx].entries[i];
#ifdef COLLIDE
      if (i > 0)
	hashtable[*indx].entries[i]->collide += i;
#endif
#define SWAP_ON_COUNT
#ifdef SWAP_ON_COUNT
      // Swap hashtable position with neighbor to the left (i.e. earlier position in the search
      // array) if we've been called more frequently
      // This should minimize the number of tests for "self" in the linear search.
      if (i > 0) {
	unsigned long neigh_count = hashtable[*indx].entries[i-1]->count;
	if (hashtable[*indx].entries[i]->count > neigh_count) {
	  Timer *tmp                    = hashtable[*indx].entries[i];
	  hashtable[*indx].entries[i]   = hashtable[*indx].entries[i-1];
	  hashtable[*indx].entries[i-1] = tmp;
	}
      }
#endif
      break;
    }
  }
  return ptr;
}

/*
** genhashidx: generate hash index
**
** Input args:
**   name: string to be hashed on
**
** Return value: hash value
*/
#define NEWWAY
static inline unsigned int genhashidx (const char *name)
{
  const unsigned char *c;       // pointer to elements of "name"
  unsigned int indx;            // return value of function
#ifdef NEWWAY
  unsigned int mididx, lastidx; // mid and final index of name

  lastidx = strlen (name) - 1;
  mididx = lastidx / 2;
#else
  int i;                        // iterator (OLDWAY only)
#endif
  // Disallow a hash index of zero (by adding 1 at the end) since user input of an 
  // uninitialized value, though an error, has a likelihood to be zero.
#ifdef NEWWAY
  c = (unsigned char *) name;
  indx = (MAX_CHARS*c[0] + (MAX_CHARS-mididx)*c[mididx] + (MAX_CHARS-lastidx)*c[lastidx]) % tablesizem1 + 1;
#else
  indx = 0;
  i = MAX_CHARS;
#pragma unroll(2)
  for (c = (unsigned char *) name; *c && i > 0; ++c) {
    indx += i*(*c);
    --i;
  }
  indx = indx % tablesizem1 + 1;
#endif

  return indx;
}

/*
** getentry: find the entry in the hash table and return a pointer to it.
**
** Input args:
**   hashtable: the hashtable (array)
**   indx:      hashtable index
**
** Return value: pointer to the entry, or NULL if not found
*/
static inline Timer *getentry (const Hashentry *hashtable, const char *name, unsigned int indx)
{
  int i;
  Timer *ptr = 0;             // return value when entry not found

  // If nument exceeds 1 there was one or more hash collisions and we must search
  // linearly through the array of names with the same hash for a match
  for (i = 0; i < hashtable[indx].nument; i++) {
    if (STRMATCH (name, hashtable[indx].entries[i]->name)) {
      ptr = hashtable[indx].entries[i];
#ifdef COLLIDE
      if (i > 0)
	hashtable[indx].entries[i]->collide += i;
#endif
#define SWAP_ON_COUNT
#ifdef SWAP_ON_COUNT
      // Swap hashtable position with neighbor to the left (i.e. earlier position in the search
      // array) if we've been called more frequently
      // This should minimize the number of tests for "name" in the linear search.
      if (i > 0) {
	unsigned long neigh_count = hashtable[indx].entries[i-1]->count;
	if (hashtable[indx].entries[i]->count > neigh_count) {
	  Timer *tmp                   = hashtable[indx].entries[i];
	  hashtable[indx].entries[i]   = hashtable[indx].entries[i-1];
	  hashtable[indx].entries[i-1] = tmp;
	}
      }
#endif
      break;
    }
  }
  return ptr;
}

/*
** Add entry points for auto-instrumented codes
** Auto instrumentation flags for various compilers:
**
** gcc, pathcc, icc: -finstrument-functions
** pgcc:             -Minstrument:functions
** xlc:              -qdebug=function_trace
*/

#ifdef _AIX
void __func_trace_enter (const char *function_name, const char *file_name, int line_number,
                         void **const user_data)
{
  if (dopr_memusage && GPTLget_thread_num() == 0)
    check_memusage ("Begin", function_name);
  (void) GPTLstart (function_name);
}
  
void __func_trace_exit (const char *function_name, const char *file_name, int line_number,
                        void **const user_data)
{
  (void) GPTLstop (function_name);
  if (dopr_memusage && GPTLget_thread_num() == 0)
    check_memusage ("End", function_name);
}
  
#else
//_AIX not defined

#if ( defined HAVE_LIBUNWIND || defined HAVE_BACKTRACE )
void __cyg_profile_func_enter (void *this_fn, void *call_site)
{
  int t;             // thread index
  int symsize;       // number of characters in symbol
  char *symnam = 0;  // symbol name whether using unwind or backtrace
  int numchars;      // number of characters in function name
  unsigned int indx; // hash table index
  Timer *ptr;        // pointer to entry if it already exists
  int ret;
  static const char *thisfunc = "__cyg_profile_func_enter";

  // In debug mode, get symbol name up front to diagnose function name
  // Otherwise live with "unknown" because get_symnam is very expensive

  // Call preamble_start rather than just GPTLget_thread_num because preamble_stop is needed for
  // other reasons in __cyg_profile_func_exit, and the preamble* functions need to mirror each
  // other.
  
  if (preamble_start (&t, unknown) != 0)
    return;
  
  ptr = getentry_instr (hashtable[t], this_fn, &indx);

  /* 
  ** Recursion => increment depth in recursion and return.  We need to return 
  ** because we don't want to restart the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */
  if (ptr && ptr->onflg) {
    ++ptr->recurselvl;
    return;
  }

  // Increment stackidx[t] unconditionally. This is necessary to ensure the correct
  // behavior when GPTLstop_instr decrements stackidx[t] unconditionally.
  if (++stackidx[t].val > MAX_STACK-1) {
    GPTLwarn ("%s: stack too big\n", thisfunc);
    return;
  }

  if ( ! ptr) {     // Add a new entry and initialize
    ptr = (Timer *) GPTLallocate (sizeof (Timer), thisfunc);
    memset (ptr, 0, sizeof (Timer));
    if (get_symnam (this_fn, &symnam) != 0) {
      printf ("%s: failed to find symbol for address %p\n", thisfunc, this_fn);
      return;
    }
    symsize = strlen (symnam);

    // For names longer than MAX_CHARS, need the full name to avoid misrepresenting
    // names with stripped off characters as duplicates
    if (symsize > MAX_CHARS) {
      ptr->longname = (char *) malloc (symsize+1);
      strcpy (ptr->longname, symnam);
    }
    numchars = MIN (symsize, MAX_CHARS);
    strncpy (ptr->name, symnam, numchars);
    ptr->name[numchars] = '\0';
    ptr->address = this_fn;
    free (symnam);

    if (update_ll_hash (ptr, t, indx) != 0) {
      GPTLwarn ("%s: update_ll_hash error\n", thisfunc);
      return;
    }
  }

  if (update_parent_info (ptr, callstack[t], stackidx[t].val) != 0) {
    GPTLwarn ("%s: update_parent_info error\n", thisfunc);
    return;
  }

  if (update_ptr (ptr, t) != 0) {
    GPTLwarn ("%s: update_ptr error\n", thisfunc);
    return;
  }

  if (dopr_memusage && t == 0)
    check_memusage ("Begin", ptr->name);
}

#ifdef HAVE_BACKTRACE
static int get_symnam (void *this_fn, char **symnam)
{
  char **strings = 0;
  void *buffer[3];
  int nptrs;
  char addrstr[MAX_CHARS+1];          // function address as a string
  static const char *thisfunc = "get_symnam(backtrace)";

  static void extract_name (char *, char **, void *);

  nptrs = backtrace (buffer, 3);
  if (nptrs != 3) {
    GPTLwarn ("%s backtrace failed nptrs should be 2 but is %d\n", thisfunc, nptrs);
    return -1;
  }

  if ( ! (strings = backtrace_symbols (buffer, nptrs))) {
    GPTLwarn ("%s backtrace_symbols failed strings is null\n", thisfunc);
    return -1;
  }

  extract_name (strings[2], symnam, this_fn);
  free (strings);
  return 0;
}

// Backtrace strings have a bunch of extra stuff in them.
// Find the start and end of the function name and return a pointer to the function name
// Note a null terminator is added to str after the name, but we don't care
void extract_name (char *str, char **symnam, void *this_fn)
{
  char *cstart;
  char *cend;
  int nchars;
  
  for (cstart = str; *cstart != '(' && *cstart != '\0'; ++cstart);
  if (*cstart == '\0') {
    cend = cstart;
  } else {
    ++cstart;
    for (cend = cstart; *cend != '+' && *cend != '\0'; ++cend);
    if (cend == cstart) {
    }
    *cend = '\0';
  }
  if (cend == cstart) {
    // Name not found: write function address into symnam. Allow 16 characters to hold address
    *symnam = (char *) malloc (16+1);
    snprintf (*symnam, 16+1,"%-16p", this_fn);
  } else {
    nchars = (int) (cend - cstart);
#ifdef APPEND_ADDRESS
    char addrname[16+2];
    *symnam = (char *) malloc (nchars+16+2);  // 16 is nchars, +2 is for '#' and '\0'
    strncpy (*symnam, cstart, nchars+1);
    snprintf (addrname, 16+2, "#%-16p", this_fn);
    strcat (*symnam, addrname);
#else
    *symnam = (char *) malloc (nchars + 1);
    strncpy (*symnam, cstart, nchars+1);
#endif
  }
}
#endif   // HAVE_BACKTRACE

#ifdef HAVE_LIBUNWIND
static int get_symnam (void *this_fn, char **symnam)
{
  char symbol[MAX_SYMBOL_NAME+1];
  unw_cursor_t cursor;
  unw_context_t context;
  unw_word_t offset, pc;
  int n;
  int nchars;
  static const char *thisfunc = "get_symnam(unwind)";

  // Initialize cursor to current frame for local unwinding.
  unw_getcontext (&context);
  unw_init_local (&cursor, &context);

  // Need to unwind 2 levels to get to function of interest
  for (n = 0; n < 2; ++n) {
    if (unw_step (&cursor) <= 0) { // unw_step failed: give up
      GPTLwarn ("%s: unw_step failed\n", thisfunc);
      return -1;
    }
  }

  unw_get_reg (&cursor, UNW_REG_IP, &pc);
  if (unw_get_proc_name (&cursor, symbol, sizeof(symbol), &offset) == 0) {
    char addrname[16+2];
    nchars = strlen (symbol);
#ifdef APPEND_ADDRESS
    *symnam = (char *) malloc (nchars+16+2);  // 16 is nchars, +2 is for '#' and '\0'
    strncpy (*symnam, symbol, nchars+1);
    snprintf (addrname, 16+2, "#%-16p", this_fn);
    strcat (*symnam, addrname);
#else
    *symnam = (char *) malloc (nchars + 1);
    strncpy (*symnam, symbol, nchars+1);
#endif
  } else {
    // Name not found: write function address into symnam. Allow 16 characters to hold address
    *symnam = (char *) malloc (16+1);
    snprintf (*symnam, 16+1, "%-16p", this_fn);
  }
  return 0;
}
#endif   // HAVE_LIBUNWIND

void __cyg_profile_func_exit (void *this_fn, void *call_site)
{
  float rss;
  int t;                     // thread index
  unsigned int indx;         // hash table index
  Timer *ptr;                // pointer to entry if it already exists
  double tp1 = 0.0;          // time stamp
  long usr = 0;              // user time (returned from get_cpustamp)
  long sys = 0;              // system time (returned from get_cpustamp)
  static const char *thisfunc = "__cyg_profile_func_exit";

  if (preamble_stop (&t, &tp1, &usr, &sys, thisfunc) != 0)
    return;
       
  ptr = getentry_instr (hashtable[t], this_fn, &indx);

  if ( ! ptr) {
    GPTLwarn ("%s: timer for %p had not been started.\n", thisfunc, this_fn);
    return;
  }

  if ( ! ptr->onflg ) {
    GPTLwarn ("%s: timer %s was already off.\n", thisfunc, ptr->name);
    return;
  }

  ++ptr->count;

  /* 
  ** Recursion => decrement depth in recursion and return.  We need to return
  ** because we don't want to stop the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */
  if (ptr->recurselvl > 0) {
    ++ptr->nrecurse;
    --ptr->recurselvl;
    return;
  }

  if (update_stats (ptr, tp1, usr, sys, t) != 0) {
    GPTLwarn ("%s: error from update_stats\n", thisfunc);
    return;
  }

  if (dopr_memusage && t == 0)
    check_memusage ("End", ptr->name);
}
#endif // HAVE_LIBUNWIND || HAVE_BACKTRACE
#endif // _AIX false branch

static void check_memusage (const char *str, const char *funcnam)
{
  float rss;

  (void) GPTLget_memusage (&rss);
  // Notify user when rss has grown by more than 1%
  if (rss > rssmax*1.01) {
    rssmax = rss;
    // Once MPI is initialized, change file pointer for process size to rank-specific file      
    set_fp_procsiz ();
    if (fp_procsiz) {
      fprintf (fp_procsiz, "%s %s rss grew to %8.2f MB\n", str, funcnam, rss);
      fflush (fp_procsiz);  // Not clear when this file needs to be closed, so flush
    } else {
      fprintf (stderr, "%s %s rss grew to %8.2f MB\n", str, funcnam, rss);
    }
  }
}

// set_fp_procsiz: Change file pointer from stderr to point to "procsiz.<rank>" once
// MPI has been initialized
static inline void set_fp_procsiz ()
{
#ifdef HAVE_LIBMPI
  int ret;
  int flag;
  static bool check_mpi_init = true; // whether to check if MPI has been init (init to true)
  char outfile[15];

  // Must only open the file once. Also more efficient to only make MPI lib inquiries once
  if (check_mpi_init) {
    ret = MPI_Initialized (&flag);
    if (flag) {
      int world_iam;
      check_mpi_init = false;
      ret = MPI_Comm_rank (MPI_COMM_WORLD, &world_iam);
      sprintf (outfile, "procsiz.%6.6d", world_iam);
      fp_procsiz = fopen (outfile, "w");
    }
  }
#endif
}

#ifdef HAVE_NANOTIME
// Copied from PAPI library
static inline long long nanotime (void)
{
  long long val = 0;
#ifdef BIT64
  do {
    unsigned int a, d;
    asm volatile ("rdtsc":"=a" (a), "=d" (d));
    (val) = ((long long) a) | (((long long) d) << 32);
  } while (0);
#else
  __asm__ __volatile__("rdtsc":"=A" (val): );
#endif
  return val;
}

#define LEN 4096

static float get_clockfreq ()
{
  FILE *fd = 0;
  char buf[LEN];
  int is;
  float freq = -1.;             // clock frequency (MHz)
  static const char *thisfunc = "get_clockfreq";
  static const char *max_freq_fn = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq";
  static const char *cpuinfo_fn = "/proc/cpuinfo";

  // First look for max_freq, but that isn't guaranteed to exist
  if ((fd = fopen (max_freq_fn, "r"))) {
    if (fgets (buf, LEN, fd)) {
      freq = 0.001 * (float) atof (buf);  // Convert from KHz to MHz
      if (verbose)
        printf ("GPTL: %s: Using max clock freq = %f for timing\n", thisfunc, freq);
    }
    clock_source = (char *) max_freq_fn;
    (void) fclose (fd);
    return freq;
  }
  
  // Next try /proc/cpuinfo. That has the disadvantage that it may give wrong info
  // for processors that have either idle or turbo mode
#ifdef HAVE_SLASHPROC
  if (verbose && freq < 0.)
    printf ("GPTL: %s: CAUTION: Can't find max clock freq. Trying %s instead\n",
            thisfunc, cpuinfo_fn);

  if ( (fd = fopen (cpuinfo_fn, "r"))) {
    while (fgets (buf, LEN, fd)) {
      if (strncmp (buf, "cpu MHz", 7) == 0) {
	for (is = 7; buf[is] != '\0' && !isdigit (buf[is]); is++);
	if (isdigit (buf[is])) {
	  freq = (float) atof (&buf[is]);
	  if (verbose)
	    printf ("GPTL: %s: Using clock freq from /proc/cpuinfo = %f for timing\n",
		    thisfunc, freq);
	  clock_source = (char *) cpuinfo_fn;
	  break;
	}
      }
    }
    (void) fclose (fd);
  }
#endif
  return freq;
}

/*
** The following are the set of underlying timing routines which may or may
** not be available. And their accompanying init routines.
** NANOTIME is currently only available on x86.
*/
static int init_nanotime ()
{
  static const char *thisfunc = "init_nanotime";
  if ((cpumhz = get_clockfreq ()) < 0)
    return GPTLerror ("%s: Can't get clock freq\n", thisfunc);

  if (verbose)
    printf ("GPTL: %s: Clock rate = %f MHz\n", thisfunc, cpumhz);

  cyc2sec = 1./(cpumhz * 1.e6);
  return 0;
}

static inline double utr_nanotime ()
{
  double timestamp = nanotime () * cyc2sec;
  return timestamp;
}
#endif

// MPI_Wtime requires MPI lib.
#ifdef HAVE_LIBMPI
static int init_mpiwtime () {return 0;}
static inline double utr_mpiwtime () {return MPI_Wtime ();}
#endif

#ifdef HAVE_LIBRT

// Probably need to link with -lrt for this one to work 
static int init_clock_gettime ()
{
  static const char *thisfunc = "init_clock_gettime";
  struct timespec tp;
  (void) clock_gettime (CLOCK_REALTIME, &tp);
  ref_clock_gettime = tp.tv_sec;
  if (verbose)
    printf ("GPTL: %s: ref_clock_gettime=%ld\n", thisfunc, (long) ref_clock_gettime);
  return 0;
}

static inline double utr_clock_gettime ()
{
  struct timespec tp;
  (void) clock_gettime (CLOCK_REALTIME, &tp);
  return (tp.tv_sec - ref_clock_gettime) + 1.e-9*tp.tv_nsec;
}
#endif

#ifdef _AIX
// High-res timer on AIX: read_real_time
static int init_read_real_time ()
{
  static const char *thisfunc = "init_read_real_time";
  timebasestruct_t ibmtime;
  (void) read_real_time (&ibmtime, TIMEBASE_SZ);
  (void) time_base_to_time (&ibmtime, TIMEBASE_SZ);
  ref_read_real_time = ibmtime.tb_high;
  if (verbose)
    printf ("GPTL: %s: ref_read_real_time=%ld\n", thisfunc, (long) ref_read_real_time);
  return 0;
}

static inline double utr_read_real_time ()
{
  timebasestruct_t ibmtime;
  (void) read_real_time (&ibmtime, TIMEBASE_SZ);
  (void) time_base_to_time (&ibmtime, TIMEBASE_SZ);
  return (ibmtime.tb_high - ref_read_real_time) + 1.e-9*ibmtime.tb_low;
}
#endif

#ifdef HAVE_GETTIMEOFDAY
// Default available most places: gettimeofday
static int init_gettimeofday ()
{
  static const char *thisfunc = "init_gettimeofday";
  struct timeval tp;
  (void) gettimeofday (&tp, 0);
  ref_gettimeofday = tp.tv_sec;
  if (verbose)
    printf ("GPTL: %s: ref_gettimeofday=%ld\n", thisfunc, (long) ref_gettimeofday);
  return 0;
}

static inline double utr_gettimeofday ()
{
  struct timeval tp;
  (void) gettimeofday (&tp, 0);
  return (tp.tv_sec - ref_gettimeofday) + 1.e-6*tp.tv_usec;
}
#endif

// placebo: does nothing and returns zero always. Useful for estimating overhead costs
static int init_placebo () {return 0;}
static inline double utr_placebo () {return (double) 0.;}

// printself_andchildren: Recurse through call tree, printing stats for self, then children
static void printself_andchildren (const Timer *ptr, FILE *fp, int t, int depth, 
				   double self_ohd, double parent_ohd, Outputfmt outputfmt)
{
  int n;

  if (depth > -1)     // -1 flag is to avoid printing stats for dummy outer timer
    printstats (ptr, fp, t, depth, true, self_ohd, parent_ohd, outputfmt);

  for (n = 0; n < ptr->nchildren; n++)
    printself_andchildren (ptr->children[n], fp, t, depth+1, self_ohd, parent_ohd, outputfmt);
}

static void print_callstack (int t, const char *caller)
{
  int idx;

  printf ("Current callstack from %s:\n", caller);
  for (idx = stackidx[t].val; idx > 0; --idx) {
    printf ("%s\n", callstack[t][idx]->name);
  }
}

// GPTLget_timersaddr: Return address of timers. NOT a public entry point
Timer **GPTLget_timersaddr () {return timers;}

#ifdef ENABLE_PMPI
/*
** GPTLgetentry: called ONLY from pmpi.c (i.e. not a public entry point). Returns a pointer to the 
**               requested timer name by calling internal function getentry()
** 
** Return value: 0 (NULL) or the return value of getentry()
*/
Timer *GPTLgetentry (const char *name)
{
  int t;
  unsigned int indx;    // returned from getentry (unused)
  static const char *thisfunc = "GPTLgetentry";

  if ( ! initialized) {
    (void) GPTLerror ("%s: initialization was not completed\n", thisfunc);
    return 0;
  }

  if ((t = GPTLget_thread_num ()) < 0) {
    (void) GPTLerror ("%s: bad return from GPTLget_thread_num\n", thisfunc);
    return 0;
  }

  indx = genhashidx (name);
  return (getentry (hashtable[t], name, indx));
}
#endif
