/**
 * @file Main file contains most user-accessible GPTL functions.
 * @Author Jim Rosinski
 */

#include "config.h" // Must be first include.
#include "private.h"
#include "gptl.h"
#include "main.h"
#include "gptl_papi.h"
#include "thread.h"
#include "memusage.h"

#ifdef HAVE_LIBMPI
#include <mpi.h>
#endif

#include <stdlib.h>        // malloc
#include <sys/time.h>      // gettimeofday
#include <sys/times.h>     // times
#include <unistd.h>        // gettimeofday, syscall
#include <stdio.h>
#include <string.h>        // memset, strcmp (via STRMATCH)

#ifdef HAVE_LIBRT
#include <time.h>
#endif

#ifdef _AIX
#include <sys/systemcfg.h>
#endif

// Local functions: Protect with "static"
    
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

static void print_callstack (int t, const char *caller)
{
  int idx;

  printf ("Current callstack from %s:\n", caller);
  for (idx = gptlmain::stackidx[t].val; idx > 0; --idx) {
    printf ("%s\n", gptlmain::callstack[t][idx]->name);
  }
}

// Start of namespace contents: most are used by other GPTL files
namespace gptlmain {
  volatile bool initialized = false;   // GPTLinitialize has been called
  volatile bool disabled = false;      // Timers disabled?
  int tablesize = DEFAULT_TABLE_SIZE;  // per-thread size of hash table (settable parameter)
  int tablesizem1 = DEFAULT_TABLE_SIZE - 1;
#ifdef HAVE_NANOTIME
  double cyc2sec = -1;                 // init to bad value
#endif
  Hashentry **hashtable;               // table of entries
  Nofalse *stackidx;                   // index into callstack:
  Timer **timers = 0;                  // linked list of timers
  Timer ***callstack;                  // call stack
  Timer **last = 0;                    // last element in timers list
  bool imperfect_nest;                 // e.g. start(A),start(B),stop(A)
  bool dopr_memusage = false;          // whether to include memusage print on growth
  Settings cpustats =      {GPTLcpu,  (char *) "     usr       sys  usr+sys", false};
  Settings wallstats =     {GPTLwall, (char *) "     Wall      max      min", true };
  int depthlimit = 99999;              // max depth for timers (99999 is effectively infinite
  bool dousepapi = false;              // saves a function call if stays false
#ifdef HAVE_GETTIMEOFDAY
  time_t ref_gettimeofday = -1;        // ref start point for gettimeofday
#endif
#ifdef HAVE_LIBRT
  time_t ref_clock_gettime = -1;       // ref start point for clock_gettime
#endif
#ifdef _AIX
  time_t ref_read_real_time = -1;      // ref start point for read_real_time
#endif

  extern "C" {
    /*
    ** genhashidx: generate hash index
    **
    ** Input args:
    **   name: string to be hashed on
    **
    ** Return value: hash value
    */
    unsigned int genhashidx (const char *name, const int namelen)
    {
      const unsigned char *c;       // pointer to elements of "name"
      unsigned int indx;            // return value of function
      unsigned int mididx, lastidx; // mid and final index of name

      lastidx = namelen - 1;
      mididx = lastidx / 2;
      // Disallow a hash index of zero (by adding 1 at the end) since user input of an 
      // uninitialized value, though an error, has a likelihood to be zero.
      c = (unsigned char *) name;
      indx = (MAX_CHARS*c[0] + (MAX_CHARS-mididx)*c[mididx] + (MAX_CHARS-lastidx)*c[lastidx])
	% gptlmain::tablesizem1 + 1;
      return indx;
    }

    double (*ptr2wtimefunc)() = 0;

    /*
    ** getentry: find the entry in the hash table and return a pointer to it.
    **
    ** Input args:
    **   hashtable: the hashtable (array)
    **   indx:      hashtable index
    **
    ** Return value: pointer to the entry, or NULL if not found
    */
    Timer *getentry (const Hashentry *hashtable, const char *name, unsigned int indx)
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

    // preamble_start: Do the things common to GPTLstart* routines
    int preamble_start (int *t, const char *name)
    {
      static const char *thisfunc = "preamble_start";

      if (gptlmain::disabled)
	return DONE;

      // Only print error message for manual instrumentation: too hard to ensure
      // GPTLinitialize() has been called for auto-instrumented code
      if ( ! gptlmain::initialized)
	return GPTLerror ("%s timername=%s: GPTLinitialize has not been called\n", thisfunc, name);
  
      if ((*t = thread::get_thread_num ()) < 0)
	return GPTLerror ("%s: bad return from GPTLget_thread_num\n", thisfunc);

      // If current depth exceeds a user-specified limit for print, just
      // increment and tell caller to return immediately (DONTSTART)
      if (gptlmain::stackidx[*t].val >= gptlmain::depthlimit) {
	++gptlmain::stackidx[*t].val;
	return DONE;
      }
      return 0;
    }

    int preamble_stop (int *t, double *tp1, long *usr, long *sys, const char *name)
    {
      static const char *thisfunc = "preamble_stop";

      if (gptlmain::disabled)
	return DONE;

      if ( ! gptlmain::initialized)
	return GPTLerror ("%s timername=%s: GPTLinitialize has not been called\n", thisfunc, name);

      // Get the wallclock timestamp
      if (gptlmain::wallstats.enabled) {
	*tp1 = (*gptlmain::ptr2wtimefunc) ();
      }

      if (gptlmain::cpustats.enabled && get_cpustamp (usr, sys) < 0)
	return GPTLerror ("%s: get_cpustamp error", name);

      if ((*t = thread::get_thread_num ()) < 0)
	return GPTLerror ("%s: bad return from GPTLget_thread_num\n", name);

      // If current depth exceeds a user-specified limit for print, just decrement and return
      if (gptlmain::stackidx[*t].val > gptlmain::depthlimit) {
	--gptlmain::stackidx[*t].val;
	return DONE;
      }
      return 0;
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
    int update_ll_hash (Timer *ptr, int t, unsigned int indx)
    {
      int nument;      // number of entries (> 0 means collision)
      Timer **eptr;    // for realloc

      gptlmain::last[t]->next = ptr;
      gptlmain::last[t] = ptr;
      ++gptlmain::hashtable[t][indx].nument;
      nument = gptlmain::hashtable[t][indx].nument;
  
      eptr = (Timer **) realloc (gptlmain::hashtable[t][indx].entries, nument * sizeof (Timer *));
      if ( ! eptr)
	return GPTLerror ("update_ll_hash: realloc error\n");
      
      gptlmain::hashtable[t][indx].entries           = eptr;
      gptlmain::hashtable[t][indx].entries[nument-1] = ptr;
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
    int update_parent_info (Timer *ptr, Timer **callstackt, int stackidxt)
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
    int update_stats (Timer *ptr, const double tp1, const long usr, const long sys,
		      const int t)
    {
      double delta;      // wallclock time difference
      int bidx;          // bottom of call stack
      Timer *bptr;       // pointer to last entry in call stack
      static const char *thisfunc = "update_stats";

      ptr->onflg = false;

#ifdef HAVE_PAPI
      if (gptlmain::dousepapi && GPTL_PAPIstop (t, &ptr->aux) < 0)
	return GPTLerror ("%s: error from GPTL_PAPIstop\n", thisfunc);
#endif

      if (gptlmain::wallstats.enabled) {
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

      if (gptlmain::cpustats.enabled) {
	ptr->cpu.accum_utime += usr - ptr->cpu.last_utime;
	ptr->cpu.accum_stime += sys - ptr->cpu.last_stime;
	ptr->cpu.last_utime   = usr;
	ptr->cpu.last_stime   = sys;
      }

      // Verify that the timer being stopped is at the bottom of the call stack
      if ( ! gptlmain::imperfect_nest) {
	char *name;        //  found name
	char *bname;       //  expected name

	bidx = gptlmain::stackidx[t].val;
	bptr = gptlmain::callstack[t][bidx];
	if (ptr != bptr) {
	  gptlmain::imperfect_nest = true;
	  if (ptr->longname)
	    name = ptr->longname;
	  else
	    name = ptr->name;
	  
	  // Print to stderr as well due to debugging importance, and warn/error have limits on the
	  // number of calls that can be made
	  fprintf (stderr, "%s thread %d: Imperfect nest detected: Got timer %s\n",
		   thisfunc, t, name);      
	  GPTLwarn ("%s thread %d: Imperfect nest detected: Got timer %s\n", thisfunc, t, name);
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
      
      --gptlmain::stackidx[t].val;           // Pop the callstack
      if (gptlmain::stackidx[t].val < -1) {
	gptlmain::stackidx[t].val = -1;
	return GPTLerror ("%s: tree depth has become negative.\n", thisfunc);
      }
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
    int update_ptr (Timer *ptr, const int t)
    {
      ptr->onflg = true;
      
      if (gptlmain::cpustats.enabled &&
	  get_cpustamp (&ptr->cpu.last_utime, &ptr->cpu.last_stime) < 0)
	return GPTLerror ("update_ptr: get_cpustamp error");
      
      if (gptlmain::wallstats.enabled) {
	double tp2 = (*gptlmain::ptr2wtimefunc) ();
	ptr->wall.last = tp2;
      }
      
#ifdef HAVE_PAPI
      if (gptlmain::dousepapi && GPTL_PAPIstart (t, &ptr->aux) < 0)
	return GPTLerror ("update_ptr: error from GPTL_PAPIstart\n");
#endif
      return 0;
    }

    // Underlying timing routines & wrappers start here
#ifdef HAVE_NANOTIME
    // Copied from PAPI library
    double utr_nanotime ()
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
      return val * gptlmain::cyc2sec;
    }
#endif

    // MPI_Wtime requires MPI lib.
#ifdef HAVE_LIBMPI
    double utr_mpiwtime () {return MPI_Wtime ();}
#endif

#ifdef _AIX
    double utr_read_real_time ()
    {
      timebasestruct_t ibmtime;
      (void) read_real_time (&ibmtime, TIMEBASE_SZ);
      (void) time_base_to_time (&ibmtime, TIMEBASE_SZ);
      return (ibmtime.tb_high - gptlmain::ref_read_real_time) + 1.e-9*ibmtime.tb_low;
    }
#endif

#ifdef HAVE_LIBRT
    double utr_clock_gettime ()
    {
      struct timespec tp;
      (void) clock_gettime (CLOCK_REALTIME, &tp);
      return (tp.tv_sec - gptlmain::ref_clock_gettime) + 1.e-9*tp.tv_nsec;
    }
#endif

#ifdef HAVE_GETTIMEOFDAY
    double utr_gettimeofday ()
    {
      struct timeval tp;
      (void) gettimeofday (&tp, 0);
      return (tp.tv_sec - gptlmain::ref_gettimeofday) + 1.e-6*tp.tv_usec;
    }
#endif

    // placebo: does nothing and returns zero always. Useful for estimating overhead costs
    double utr_placebo () {return (double) 0.;}
    
#ifdef ENABLE_NESTEDOMP
    inline void get_nested_thread_nums (int *major, int *minor)
    {
      if (omp_get_max_active_levels () > 1) { // nesting is "enabled", though not necessarily active
	volatile const int lvl = omp_get_active_level (); // lvl=2 => inside 2 #pragma omp regions
	if (lvl == 1) {
	  *major = omp_get_thread_num ();
	} else if (lvl == 2) {
	  *major = omp_get_ancestor_thread_num (1);
	  *minor = omp_get_thread_num ();
	}
      }
    }
#endif
  }
}

// VERBOSE is a debugging ifdef local to the rest of this file
#undef VERBOSE
// APPEND_ADDRESS is an autoprofiling debugging ifdef that will append function address to name
#undef APPEND_ADDRESS

// Start of user-callable functions
/*
** GPTLstart: start a timer
**
** Input arguments:
**   name: timer name
**
** Return value: 0 (success) or GPTLerror (failure)
*/
extern "C" int GPTLstart (const char *name, int namelen=-1)
{
  Timer *ptr;
  int t;
  int ret;
  unsigned int indx; // index into hash table
  static const char *thisfunc = "GPTLstart";
  
  ret = gptlmain::preamble_start (&t, name);
  if (ret == DONE)
    return 0;
  else if (ret != 0)
    return ret;
  
  if (namelen < 0)
    namelen = strlen (name);
  if (namelen > MAX_CHARS)
    return GPTLerror ("%s: region name %s is too long\nRename to be %d chars or fewer\n",
		      thisfunc, name, MAX_CHARS);
  
  // ptr will point to the requested timer in the current list, or NULL if this is a new entry
  indx = gptlmain::genhashidx (name, namelen);
  ptr = gptlmain::getentry (gptlmain::hashtable[t], name, indx);

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
  if (++gptlmain::stackidx[t].val > MAX_STACK-1)
    return GPTLerror ("%s: stack too big: NOT starting timer for %s\n", thisfunc, name);

  if ( ! ptr) {   // Add a new entry and initialize. longname only needed for auto-profiling
    ptr = (Timer *) GPTLallocate (sizeof (Timer), thisfunc);
    memset (ptr, 0, sizeof (Timer));
    strncpy (ptr->name, name, namelen);
    ptr->name[namelen] = '\0';

    if (gptlmain::update_ll_hash (ptr, t, indx) != 0)
      return GPTLerror ("%s: update_ll_hash error\n", thisfunc);
  }

  if (gptlmain::update_parent_info (ptr, gptlmain::callstack[t], gptlmain::stackidx[t].val) != 0)
    return GPTLerror ("%s: update_parent_info error\n", thisfunc);

  if (gptlmain::update_ptr (ptr, t) != 0)
    return GPTLerror ("%s: update_ptr error\n", thisfunc);

#ifdef ENABLE_NESTEDOMP
  get_nested_thread_nums (&ptr->major, &ptr->minor);
#endif

  if (gptlmain::dopr_memusage && t == 0)
    memusage::check_memusage ("Begin", ptr->name);

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
extern "C" int GPTLinit_handle (const char *name, int *handle, int namelen=-1)
{
  if (gptlmain::disabled)
    return 0;

  if (namelen < 0)
    namelen = strlen (name);

  *handle = (int) gptlmain::genhashidx (name, namelen);
  return 0;
}

/*
** GPTLstart_handle: start a timer based on a handle
**
** Input arguments:
**   name:   timer name (handle=0 on input means generate handle from name)
**
** Input/output arguments:
**   handle: zero means generate from name. Non-zero means value was pre-generated
**
** Return value: 0 (success) or GPTLerror (failure)
*/
extern "C" int GPTLstart_handle (const char *name, int *handle, int namelen=-1)
{
  Timer *ptr;
  int t;
  int ret;
  static const char *thisfunc = "GPTLstart_handle";

  ret = gptlmain::preamble_start (&t, name);
  if (ret == DONE)
    return 0;
  else if (ret != 0)
    return ret;

  if (namelen < 0)
    namelen = strlen (name);
  if (namelen > MAX_CHARS)
    return GPTLerror ("%s: region name %s is too long\nRename to be %d chars or fewer\n",
		      thisfunc, name, MAX_CHARS);
  
  /*
  ** If handle is zero on input, generate the hash entry and return it to the user.
  ** Otherwise assume it's a previously generated hash index passed in by the user.
  ** Don't need a critical section here--worst case multiple threads will generate the
  ** same handle and store to the same memory location, and this will only happen once.
  */
  if (*handle == 0) {
    *handle = (int) gptlmain::genhashidx (name, namelen);
#ifdef VERBOSE
    printf ("%s: name=%s thread %d generated handle=%d\n", thisfunc, name, t, *handle);
#endif
  } else if ((unsigned int) *handle > gptlmain::tablesizem1) {
    return GPTLerror ("%s: Bad input handle=%u exceeds tablesizem1=%d\n", 
		      thisfunc, (unsigned int) *handle, gptlmain::tablesizem1);
  }

  ptr = gptlmain::getentry (gptlmain::hashtable[t], name, (unsigned int) *handle);
  
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
  if (++gptlmain::stackidx[t].val > MAX_STACK-1)
    return GPTLerror ("%s: stack too big: NOT starting timer for %s\n", thisfunc, name);

  if ( ! ptr) { // Add a new entry and initialize
    // Verify *handle matches what genhashidx says (only useful when GPTLinit_handle called)
    int testidx = (int) gptlmain::genhashidx (name, namelen);
    if (testidx != *handle)
      return GPTLerror ("%s: expected vs. input handles for name=%s don't match.",
			" Possible user error passing wrong handle for name\n",
			thisfunc, name);
    
    ptr = (Timer *) GPTLallocate (sizeof (Timer), thisfunc);
    memset (ptr, 0, sizeof (Timer));
    strncpy (ptr->name, name, namelen);
    ptr->name[namelen] = '\0';

    if (gptlmain::update_ll_hash (ptr, t, (unsigned int) *handle) != 0)
      return GPTLerror ("%s: update_ll_hash error\n", thisfunc);
  }

  if (gptlmain::update_parent_info (ptr, gptlmain::callstack[t], gptlmain::stackidx[t].val) != 0)
    return GPTLerror ("%s: update_parent_info error\n", thisfunc);

  if (gptlmain::update_ptr (ptr, t) != 0)
    return GPTLerror ("%s: update_ptr error\n", thisfunc);

#ifdef ENABLE_NESTEDOMP
  get_nested_thread_nums (&timers[t]->major, &timers[t]->minor);
#endif

  if (gptlmain::dopr_memusage && t == 0)
    memusage::check_memusage ("Begin", ptr->name);

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
extern "C" int GPTLstop (const char *name, int namelen=-1)
{
  double tp1 = 0.0;          // wallclock time stamp
  Timer *ptr;
  int t;
  int ret;
  unsigned int indx;         // hash indexx
  long usr = 0;              // user time (returned from get_cpustamp)
  long sys = 0;              // system time (returned from get_cpustamp)
  static const char *thisfunc = "GPTLstop";

  ret = gptlmain::preamble_stop (&t, &tp1, &usr, &sys, name);
  if (ret == DONE)
    return 0;
  else if (ret != 0)
    return ret;
       
  if (namelen < 0)
    namelen = strlen (name);
    
  indx = gptlmain::genhashidx (name, namelen);
  if (! (ptr = gptlmain::getentry (gptlmain::hashtable[t], name, indx)))
    return GPTLerror ("%s thread %d: timer %s had not been started.\n"
  		      "Perhaps length exceeds %d chars?\n", thisfunc, t, name, MAX_CHARS);

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

  if (gptlmain::update_stats (ptr, tp1, usr, sys, t) != 0)
    return GPTLerror ("%s: error from update_stats\n", thisfunc);

  if (gptlmain::dopr_memusage && t == 0)
    memusage::check_memusage ("End", ptr->name);

  return 0;
}

/*
** GPTLstop_handle: stop a timer based on a handle
**
** Input arguments:
**   name:   timer name
**   handle: previously generated handle (= ehash index). Pointer not int for consistency w/start
**
** Return value: 0 (success) or -1 (failure)
*/
extern "C" int GPTLstop_handle (const char *name, int *handle, int namelen=-1)
{
  double tp1 = 0.0;          // wallclock time stamp
  Timer *ptr;
  int t;
  int ret;
  long usr = 0;              // user time (returned from get_cpustamp)
  long sys = 0;              // system time (returned from get_cpustamp)
  unsigned int indx;         // index into hash table
  static const char *thisfunc = "GPTLstop_handle";

  ret = gptlmain::preamble_stop (&t, &tp1, &usr, &sys, thisfunc);
  if (ret == DONE)
    return 0;
  else if (ret != 0)
    return ret;
       
  indx = (unsigned int) *handle;
  if (indx == 0 || indx > gptlmain::tablesizem1) 
    return GPTLerror ("%s: bad input handle=%u for timer %s.\n", thisfunc, indx, name);
  
  if ( ! (ptr = gptlmain::getentry (gptlmain::hashtable[t], name, indx)))
    return GPTLerror ("%s: handle=%u has not been set for timer %s.\n"
  		      "Perhaps length exceeds %d chars?\n", thisfunc, indx, name, MAX_CHARS);
  if ( ! ptr->onflg )
    return GPTLerror ("%s: timer %s was already off.\n", thisfunc, ptr->name);

  ++ptr->count;

  // On first call, verify *handle matches what genhashidx says
  if (ptr->count == 1) {
    if (namelen < 0)
      namelen = strlen (name);
    
    int testidx = (int) gptlmain::genhashidx (name, namelen);
    if (testidx != *handle)
      return GPTLerror ("%s: expected vs. input handles for name=%s don't match.",
			" Possible user error passing wrong handle for name\n",
			thisfunc, name);
  }
	
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

  if (gptlmain::update_stats (ptr, tp1, usr, sys, t) != 0)
    return GPTLerror ("%s: error from update_stats\n", thisfunc);

  if (gptlmain::dopr_memusage && t == 0)
    memusage::check_memusage ("End", ptr->name);

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
extern "C" int GPTLstartstop_val (const char *name, double value, int namelen=-1)
{
  Timer *ptr;
  int t;
  unsigned int indx;         // index into hash table
  static const char *thisfunc = "GPTLstartstop_val";

  if (gptlmain::disabled)
    return 0;

  if ( ! gptlmain::initialized)
    return GPTLerror ("%s: GPTLinitialize has not been called\n", thisfunc);

  if ( ! gptlmain::wallstats.enabled)
    return GPTLerror ("%s: wallstats must be enabled to call this function\n", thisfunc);

  if (value < 0.)
    return GPTLerror ("%s: Input value must not be negative\n", thisfunc);

  if ((t = thread::get_thread_num ()) < 0)
    return GPTLerror ("%s: bad return from GPTLget_thread_num\n", thisfunc);

  if (namelen < 0)
    namelen = strlen (name);
    
  // Find out if the timer already exists
  indx = gptlmain::genhashidx (name, namelen);
  ptr = gptlmain::getentry (gptlmain::hashtable[t], name, indx);

  if (ptr) {
    // The timer already exists. Bump the count manually, update the time stamp,
    // and let control jump to the point where wallclock settings are adjusted.
    ++ptr->count;
    ptr->wall.last = (*gptlmain::ptr2wtimefunc) ();
  } else {
    // Need to call start/stop to set up linked list and hash table.
    // "count" and "last" will also be set properly by the call to this pair.
    if (GPTLstart (name, namelen) != 0)
      return GPTLerror ("%s: Error from GPTLstart\n", thisfunc);

    if (GPTLstop (name, namelen) != 0)
      return GPTLerror ("%s: Error from GPTLstop\n", thisfunc);

    // start/stop pair just called should guarantee ptr will be found
    if ( ! (ptr = gptlmain::getentry (gptlmain::hashtable[t], name, indx)))
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

// If specified at configure time, insert appropriate threading file instead of compiling
// separately so that GPTLget_thread_num may be inlined.

#ifdef INLINE_THREADING
#if ( defined UNDERLYING_OPENMP )
#include "./thread_omp.cc"
#elif ( defined UNDERLYING_PTHREADS )
#include "./thread_pthreads.cc"
#else
#include "./thread_none.cc"
#endif
#endif
