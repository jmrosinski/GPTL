#include "config.h" // must be 1st include
#include "once.h"
#include "thread.h"
#include "util.h"
#include "postprocess.h"
#include "private.h"
#include "gptl.h"   // user-visible prototypes

#include <stdio.h>
#include <ctype.h>         // isdigit
#include <time.h>      // time_t
#include <stdlib.h>        // atof
#include <string.h>
#include <unistd.h>    // sysconf

// Local variables to the file
static const int LEN = 4096;

// Namespace declarations of data/functions to be used only by GPTL
namespace gptl_once {
  float cpumhz = -1.;                 // clock freq. in mhz. Init to bad value
  volatile bool initialized = false;  // GPTLinitialize has been called
  bool percent = false;               // print wallclock also as percent of 1st timers[0]
  bool dopr_preamble = true;          // whether to print preamble info
  bool dopr_threadsort = true;        // whether to print sorted thread stats
  bool dopr_multparent = true;        // whether to print multiple parent info
  bool dopr_collision = false;        // whether to print hash collision info
  bool dopr_memusage = false;         // whether to include memusage print when auto-profiling
  bool verbose = false;               // output verbosity
  long ticks_per_sec;                 // clock ticks per second
  GPTL_Method method = GPTLmost_frequent;  // default parent/child printing mechanism
  int depthlimit = 99999;             // max depth for timers (99999 is effectively infinite)
  bool sync_mpi = false;              // auto-synchronize MPI calls when PMPI enabled
  bool onlypr_rank0 = false;          // flag says only print from MPI rank 0 (default false)
  time_t ref_gettimeofday = -1;       // ref start point for gettimeofday
  time_t ref_clock_gettime = -1;      // ref start point for clock_gettime
  int funcidx = 0;                    // default timer is gettimeofday
#ifdef HAVE_NANOTIME
  double cyc2sec = -1;                // convert cycles to seconds. init to bad value
  char *clock_source = unknown;       // where clock found
#endif

  extern "C" {
    Funcentry funclist[] = {
#ifdef HAVE_GETTIMEOFDAY
      {GPTLgettimeofday,   gptl_private::utr_gettimeofday,   init_gettimeofday,  "gettimeofday"},
#endif
#ifdef HAVE_NANOTIME
      {GPTLnanotime,       gptl_private::utr_nanotime,       init_nanotime,      "nanotime"},
#endif
#ifdef HAVE_LIBMPI
      {GPTLmpiwtime,       gptl_private::utr_mpiwtime,       init_mpiwtime,      "MPI_Wtime"},
#endif
#ifdef HAVE_LIBRT
      {GPTLclockgettime,   gptl_private::utr_clock_gettime,  init_clock_gettime, "clock_gettime"},
#endif
#ifdef _AIX
      {GPTLread_real_time, gptl_private::utr_read_real_time, init_read_real_time,"read_real_time"},
#endif
      {GPTLplacebo,        gptl_private::utr_placebo,        init_placebo,       "placebo"}
    };
    const int nfuncentries = sizeof (funclist) / sizeof (Funcentry);
    // The following are the set of initializers for underlying timing routines which may or may
    // not be available. NANOTIME is only available on x86.
#ifdef HAVE_NANOTIME
    int init_nanotime ()
    {
      using namespace gptl_util;
      static const char *thisfunc = "init_nanotime";
      if ((cpumhz = get_clockfreq ()) < 0)
	return error ("%s: Can't get clock freq\n", thisfunc);

      if (verbose)
	printf ("GPTL: %s: Clock rate = %f MHz\n", thisfunc, cpumhz);

      cyc2sec = 1./(cpumhz * 1.e6);
      return 0;
    }
#endif

    namespace {
      float get_clockfreq ()
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

#ifdef HAVE_LIBMPI
      int init_mpiwtime ()
      {
	return 0;
      }
#endif

      // Probably need to link with -lrt for this one to work 
#ifdef HAVE_LIBRT
      int init_clock_gettime ()
      {
	static const char *thisfunc = "init_clock_gettime";
	struct timespec tp;
	(void) clock_gettime (CLOCK_REALTIME, &tp);
	ref_clock_gettime = tp.tv_sec;
	if (verbose)
	  printf ("GPTL: %s: ref_clock_gettime=%ld\n", thisfunc, (long) ref_clock_gettime);
	return 0;
      }
#endif

      // High-res timer on AIX: read_real_time
#ifdef _AIX
      int init_read_real_time ()
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
#endif

      // Default available most places: gettimeofday
#ifdef HAVE_GETTIMEOFDAY
      int init_gettimeofday ()
      {
	static const char *thisfunc = "init_gettimeofday";
	struct timeval tp;
	(void) gettimeofday (&tp, 0);
	ref_gettimeofday = tp.tv_sec;
	if (verbose)
	  printf ("GPTL: %s: ref_gettimeofday=%ld\n", thisfunc, (long) ref_gettimeofday);
	return 0;
      }
#endif

      // placebo: does nothing and returns zero always. Useful for estimating overhead costs
      int init_placebo ()
      {
	return 0;
      }

      // set_abort_on_error: Set abort_on_error flag
      void set_abort_on_error (bool val)
      {
	using namespace gptl_util;
	abort_on_error = val;
      }
    }
  }
}

extern "C" {
  // Public entry points
  /**
   * Set option value
   *
   * @param option option to be set
   * @param val value to which option should be set (nonzero=true, zero=false)
   *
   * @return: 0 (success) or gptl_util::error (failure)
   */
  int GPTLsetoption (const int option, const int val)
  {
    using namespace gptl_once;
    using namespace gptl_util;
    using namespace gptl_thread;
    
    static const char *thisfunc = "GPTLsetoption";
      
    if (initialized)
      return gptl_util::error ("%s: must be called BEFORE GPTLinitialize\n", thisfunc);
      
    if (option == GPTLabort_on_error) {
      set_abort_on_error ((bool) val);
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
	return gptl_util::error ("%s: times() not available\n", thisfunc);
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
      method = (GPTL_Method) val; 
      if (verbose)
	printf ("%s: print_method = %s\n", thisfunc, methodstr (method));
      return 0;
    case GPTLtablesize:
      if (val < 1)
	return gptl_util::error ("%s: tablesize must be positive. %d is invalid\n", thisfunc, val);
      tablesize = val;
      tablesizem1 = val - 1;
      if (verbose)
	printf ("%s: tablesize = %d\n", thisfunc, tablesize);
      return 0;
    case GPTLsync_mpi:
#ifdef ENABLE_PMPI
      sync_mpi = (bool) val;
      if (verbose)
	printf ("%s: boolean sync_mpi = %d\n", thisfunc, val);
#else
      fprintf (stderr, "%s: option GPTLsync_mpi requires MPI\n", thisfunc);
#endif
      return 0;
    case GPTLmaxthreads:
      if (val < 1)
	return error ("%s: maxthreads must be positive. %d is invalid\n", thisfunc, val);

      maxthreads = val;
      return 0;
    case GPTLonlyprint_rank0:
      onlypr_rank0 = (bool) val; 
      if (verbose)
	printf ("%s: onlypr_rank0 = %d\n", thisfunc, val);
      return 0;
    
    case GPTLmultiplex:
      /* Allow GPTLmultiplex to fall through because it will be handled by GPTL_PAPIsetoption() */
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

    return gptl_util::error ("%s: failure to enable option %d\n", thisfunc, option);
  }

  /*
  ** GPTLsetutr: set underlying timing routine.
  **
  ** Input arguments:
  **   option: index which sets function
  **
  ** Return value: 0 (success) or gptl_util::error (failure)
  */
  int GPTLsetutr (const int option)
  {
    using namespace gptl_once;
    using namespace gptl_util;
    int i;  // index over number of underlying timer
    static const char *thisfunc = "GPTLsetutr";

    if (initialized)
      return error ("%s: must be called BEFORE GPTLinitialize\n", thisfunc);

    for (i = 0; i < nfuncentries; i++) {
      if (option == (int) funclist[i].option) {
	if (verbose)
	  printf ("%s: underlying wallclock timer = %s\n", thisfunc, funclist[i].name);
	funcidx = i;

	/*
	** Return an error condition if the function is not available.
	** OK for the user code to ignore: GPTLinitialize() will reset to gettimeofday
	*/

	if ((*funclist[i].funcinit)() < 0)
	  return error ("%s: utr=%s not available or doesn't work\n", thisfunc, funclist[i].name);
	else
	  return 0;
      }
    }
    return error ("%s: unknown option %d\n", thisfunc, option);
  }

  /*
  ** GPTLinitialize (): Initialization routine must be called from single-threaded
  **   region before any other timing routines may be called.  The need for this
  **   routine could be eliminated if not targetting timing library for threaded
  **   capability. 
  **
  ** return value: 0 (success) or gptl_util::error (failure)
  */
  int GPTLinitialize (void)
  {
    using namespace gptl_once;
    using namespace gptl_util;
    using namespace gptl_thread;
    using namespace gptl_private;
    int i;          // loop index
    int t;          // thread index
    double t1, t2;  // returned from underlying timer
    static const char *thisfunc = "GPTLinitialize";

    if (initialized)
      return error ("%s: has already been called\n", thisfunc);

    if (threadinit () < 0)
      return error ("%s: bad return from threadinit\n", thisfunc);

    if ((ticks_per_sec = sysconf (_SC_CLK_TCK)) == -1)
      return error ("%s: failure from sysconf (_SC_CLK_TCK)\n", thisfunc);

    // Allocate space for global arrays
    callstack       = (Timer ***)    allocate (maxthreads * sizeof (Timer **), thisfunc);
    stackidx        = (Nofalse *)    allocate (maxthreads * sizeof (Nofalse), thisfunc);
    timers          = (Timer **)     allocate (maxthreads * sizeof (Timer *), thisfunc);
    last            = (Timer **)     allocate (maxthreads * sizeof (Timer *), thisfunc);
    hashtable       = (Hashentry **) allocate (maxthreads * sizeof (Hashentry *), thisfunc);

    // Initialize array values
    for (t = 0; t < maxthreads; t++) {
      callstack[t] = (Timer **) allocate (MAX_STACK * sizeof (Timer *), thisfunc);
      hashtable[t] = (Hashentry *) allocate (tablesize * sizeof (Hashentry), thisfunc);
      for (i = 0; i < tablesize; i++) {
	hashtable[t][i].nument = 0;
	hashtable[t][i].entries = (E_array *) allocate (1 * sizeof (E_array), thisfunc);
      }

      // Make a timer "GPTL_ROOT" to ensure no orphans, and to simplify printing
      timers[t] = (Timer *) allocate (sizeof (Timer), thisfunc);
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
    if (PAPIinitialize (maxthreads, verbose, &GPTLnevents, GPTLeventlist) < 0)
      return error ("%s: Failure from PAPIinitialize\n", thisfunc);
#endif

    // Call init routine for underlying timing routine
    if ((*funclist[funcidx].funcinit)() < 0) {
      fprintf (stderr, "%s: Failure initializing %s. Reverting underlying timer to %s\n", 
	       thisfunc, funclist[funcidx].name, funclist[0].name);
      funcidx = 0;
    }

    gptl_private::ptr2wtimefunc = funclist[funcidx].func;

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
  ** return value: 0 (success) or error (failure)
  */
  int GPTLfinalize (void)
  {
    using namespace gptl_private;
    using namespace gptl_util;
    using namespace gptl_once;
    using namespace gptl_thread;
    int t;                // thread index
    int n;                // array index
    Timer *ptr, *ptrnext; // linked list indices
    static const char *thisfunc = "GPTLfinalize";

    if ( ! initialized)
      return error ("%s: initialization was not completed\n", thisfunc);

    for (t = 0; t < maxthreads; ++t) {
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
	delete ptr;
      }
    }

    free (callstack);
    free (stackidx);
    free (timers);
    free (last);
    free (hashtable);

    threadfinalize ();
    (void) GPTLreset_errors ();

#ifdef HAVE_PAPI
    PAPIfinalize (maxthreads);
#endif

    // Reset initial values
    timers = 0;
    last = 0;
    nthreads = -1;
#ifdef THREADED_PTHREADS
    maxthreads = MAX_THREADS;
#else
    maxthreads = -1;
#endif
    depthlimit = 99999;
    gptl_private::disabled = false;
    initialized = false;
    dousepapi = false;
    verbose = false;
    percent = false;
    dopr_preamble = true;
    dopr_threadsort = true;
    dopr_multparent = true;
    dopr_collision = true;
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
}
