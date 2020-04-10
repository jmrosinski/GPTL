/**
 * @file Contains start/stop functions and their sub-functions
 * @Author Jim Rosinski
 */

#include "config.h" // Must be first include.
#include "private.h"
#include "gptl.h"
#include "once.h"
#include "thread.h"
#include "util.h"
#include "gptl_papi.h"

#ifdef HAVE_TIMES
#include <sys/times.h>
#endif

#ifdef HAVE_LIBMPI
#include <mpi.h>
#endif

#include <stdio.h>
#include <string.h>        // memset, strcmp (via STRMATCH)
#include <stdlib.h>

#define DONE 1

extern "C" {
  //*************************************************************************************
  // User-visible functions: need to be outside namespace for callability from C, Fortran
  //*************************************************************************************

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
    using namespace gptl_once;
    using namespace gptl_private;
    using namespace gptl_util;
    Timer *ptr;        // linked list entry
    int t;             // thread index (of this thread)
    int ret;           // return value
    int numchars;      // number of characters to copy
    unsigned int indx; // hash table index
    static const char *thisfunc = "GPTLstart";
  
    ret = gptl_private::preamble_start (&t, name);
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
    if (stackidx[t].val++ > MAX_STACK-1)
      return error ("%s: stack too big: NOT starting timer for %s\n", thisfunc, name);

    if ( ! ptr) {   // New entry. Pass NULL for longname when not auto-instrumented
      ptr = new Timer (name, NULL);
      if (update_ll_hash (ptr, t, indx) != 0)
	return error ("%s: update_ll_hash error\n", thisfunc);
    }

    if (update_parent_info (ptr, callstack[t], stackidx[t].val) != 0)
      return error ("%s: update_parent_info error\n", thisfunc);

    if (update_ptr (ptr, t) != 0)
      return error ("%s: update_ptr error\n", thisfunc);
    
    if (dopr_memusage && t == 0)
      check_memusage ("Begin", ptr->name);

    return 0;
  }

  /*
  ** GPTLinit_handle: Initialize a handle for use by GPTLstart_handle() and GPTLstop_handle()
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
    using namespace gptl_private;
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
  ** Return value: 0 (success) or gptl_util::error (failure)
  */
  int GPTLstart_handle (const char *name, int *handle)
  {
    using namespace gptl_private;
    using namespace gptl_util;
    Timer *ptr;    // linked list pointer
    int t;         // thread index (of this thread)
    int ret;       // return value
    int numchars;  // number of characters to copy
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
    } else if ((unsigned int) *handle > gptl_private::tablesizem1) {
      return error ("%s: Bad input handle=%u exceeds tablesizem1=%d\n", 
		    thisfunc, (unsigned int) *handle, gptl_private::tablesizem1);
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
      return gptl_util::error ("%s: stack too big: NOT starting timer for %s\n", thisfunc, name);

    if ( ! ptr) { // Add a new entry and initialize
      ptr = new Timer (name, NULL);

      if (update_ll_hash (ptr, t, (unsigned int) *handle) != 0)
	return error ("%s: update_ll_hash error\n", thisfunc);
    }

    if (update_parent_info (ptr, callstack[t], stackidx[t].val) != 0)
      return error ("%s: update_parent_info error\n", thisfunc);

    if (update_ptr (ptr, t) != 0)
      return error ("%s: update_ptr error\n", thisfunc);

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
    using namespace gptl_private;
    using namespace gptl_util;
    using namespace gptl_once;
    Timer *ptr;                // linked list pointer
    int t;                     // thread number for this process
    double tp1 = 0.0;          // time stamp
    long usr = 0;              // user time (returned from get_cpustamp)
    long sys = 0;              // system time (returned from get_cpustamp)
    int ret;                   // return value
    unsigned int indx;         // index into hash table
    static const char *thisfunc = "GPTLstop";

    ret = preamble_stop (&t, &tp1, &usr, &sys, name);
    if (ret == DONE)
      return 0;
    else if (ret != 0)
      return ret;
       
    indx = genhashidx (name);
    if (! (ptr = getentry (hashtable[t], name, indx)))
      return error ("%s thread %d: timer for %s had not been started.\n", thisfunc, t, name);

    if ( ! ptr->onflg )
      return error ("%s: timer %s was already off.\n", thisfunc, ptr->name);

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
      return error ("%s: error from update_stats\n", thisfunc);

    if (dopr_memusage && t == 0)
      check_memusage ("End", ptr->name);

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
    using namespace gptl_private;
    using namespace gptl_util;
    using namespace gptl_once;
    double tp1 = 0.0;          // time stamp
    Timer *ptr;                // linked list pointer
    int t;                     // thread number for this process
    int ret;                   // return value
    long usr = 0;              // user time (returned from get_cpustamp)
    long sys = 0;              // system time (returned from get_cpustamp)
    unsigned int indx;
    static const char *thisfunc = "GPTLstop_handle";

    ret = preamble_stop (&t, &tp1, &usr, &sys, thisfunc);
    if (ret == DONE)
      return 0;
    else if (ret != 0)
      return ret;
       
    indx = (unsigned int) *handle;
    if (indx == 0 || indx > gptl_private::tablesizem1) 
      return error ("%s: bad input handle=%u for timer %s.\n", thisfunc, indx, name);
  
    if ( ! (ptr = getentry (hashtable[t], name, indx)))
      return error ("%s: handle=%u has not been set for timer %s.\n", thisfunc, indx, name);

    if ( ! ptr->onflg )
      return error ("%s: timer %s was already off.\n", thisfunc, ptr->name);

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
      return error ("%s: error from update_stats\n", thisfunc);

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
    using namespace gptl_private;
    using namespace gptl_once;
    using namespace gptl_thread;
    using namespace gptl_util;
    Timer *ptr;                // linked list pointer
    int t;                     // thread number for this process
    unsigned int indx;         // index into hash table
    static const char *thisfunc = "GPTLstartstop_val";

    if (disabled)
      return 0;

    if ( ! initialized)
      return error ("%s: GPTLinitialize has not been called\n", thisfunc);

    if ( ! wallstats.enabled)
      return error ("%s: wallstats must be enabled to call this function\n", thisfunc);

    if (value < 0.)
      return error ("%s: Input value must not be negative\n", thisfunc);

    // gptl_private::getentry requires the thread number
    if ((t = get_thread_num ()) < 0)
      return error ("%s: bad return from get_thread_num\n", thisfunc);

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
	return error ("%s: Error from GPTLstart\n", thisfunc);

      if (GPTLstop (name) != 0)
	return error ("%s: Error from GPTLstop\n", thisfunc);

      // start/stop pair just called should guarantee ptr will be found
      if ( ! (ptr = getentry (hashtable[t], name, indx)))
	return error ("%s: Unexpected error from getentry\n", thisfunc);

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
}

// Private to GPTL but visible across files

namespace gptl_private {
  extern "C" {
    // Unfortunately can't use multiple constructors due to extern "C" requirement,
    // so non-instrumented callers must pass in NULL for the last 2 args.
    // symnam and this_fn are non-NULL only when auto-instrumented
    // this_fn is only non-NULL when size exceeds MAX_CHARS 
    Timer::Timer (const char *name, void *this_fn)
    {
      int numchars;
      int symsize;
      
      memset (this, 0, sizeof (Timer));
      symsize = strlen (name);
      numchars = MIN (symsize, MAX_CHARS);
      strncpy (this->name, name, numchars);
      this->name[numchars] = '\0';
      
      if (this_fn) {
	this->address = this_fn;
	// For names longer than MAX_CHARS, need the full name to avoid misrepresenting
	// names with stripped off characters as duplicates
	if (symsize > MAX_CHARS) {
	  this->longname = (char *) malloc (symsize+1);
	  strcpy (this->longname, name);
	}
      }
    }
  }

  bool disabled = false;
  bool dousepapi = false;         // saves a function call if stays false
  char unknown[] = "unknown";
  Timer **timers = 0;             // linked list of timers
  Timer **last = 0;               // last element in list

  // Options, print strings, and default enable flags
  //                        user flag     string for printing            default 
  Settings cpustats =      {GPTLcpu,      "     usr       sys  usr+sys", false};
  Settings wallstats =     {GPTLwall,     "     Wall      max      min", true };
  Settings overheadstats = {GPTLoverhead, "   selfOH parentOH"         , true };

  Hashentry **hashtable = 0;    // table of entries
  Timer ***callstack = 0;       // call stack
  Nofalse *stackidx = 0;        // index into callstack
  
  int tablesize = DEFAULT_TABLE_SIZE;  // per-thread size of hash table (settable parameter)
  int tablesizem1 = DEFAULT_TABLE_SIZE - 1;
  float rssmax = 0;                 // max rss of the process
  bool imperfect_nest;              // e.g. start(A),start(B),stop(A)
  FILE *fp_procsiz = NULL;          // process size file pointer: init to 0 to use stderr

  // VERBOSE is a debugging ifdef local to the rest of this file
#undef VERBOSE

  // All user-callable functions need C linkage due to calling from C or Fortran.
  // But make all functions use C linkage to avoid cascade issues falling into private routines
  extern "C" {
    double (*ptr2wtimefunc)() = NULL;    // The underlying timing routine: init to invalid

    /*
    ** preamble_start: Do the things common to GPTLstart* routines
    **
    ** Input arguments:
    **   name: timer name
    **
    ** Output arguments:
    **   t: thread number
    **
    ** Return value: 0 (success) or GPTLerror (failure)
    */
    inline int preamble_start (int *t, const char *name)
    {
      using namespace gptl_once;
      using namespace gptl_util;
      using namespace gptl_thread;
      static const char *thisfunc = "preamble_start";
      
      if (disabled)
	return DONE;

      if ( ! initialized)
	return error ("%s timername=%s: GPTLinitialize has not been called\n", thisfunc, name);
  
      if ((*t = get_thread_num ()) < 0)
	return error ("%s: bad return from get_thread_num\n", thisfunc);

      // If current depth exceeds a user-specified limit for print, just
      // increment and tell caller to return immediately (DONTSTART)
      if (stackidx[*t].val >= depthlimit) {
	++stackidx[*t].val;
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
      int nument;      // number of entries
      Timer **eptr;    // for realloc

      last[t]->next = ptr;
      last[t] = ptr;
      ++hashtable[t][indx].nument;
      nument = hashtable[t][indx].nument;
  
      eptr = (Timer **) realloc (hashtable[t][indx].entries, nument * sizeof (Timer *));
      if ( ! eptr)
	return gptl_util::error ("update_ll_hash: realloc error\n");

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
    ** Return value: 0 (success) or gptl_util::error (failure)
    */
    inline int update_ptr (Timer *ptr, const int t)
    {
      ptr->onflg = true;

      if (cpustats.enabled && get_cpustamp (&ptr->cpu.last_utime, &ptr->cpu.last_stime) < 0)
	return gptl_util::error ("update_ptr: get_cpustamp error");
  
      if (wallstats.enabled) {
	double tp2 = (*ptr2wtimefunc) ();  // get the timestamp
	ptr->wall.last = tp2;
      }

#ifdef HAVE_PAPI
      if (dousepapi && gptl_papi::PAPIstart (t, &ptr->aux) < 0)
	return gptl_util::error ("update_ptr: error from GPTL_PAPIstart\n");
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
    ** Return value: 0 (success) or gptl_util::error (failure)
    */
    inline int update_parent_info (Timer *ptr, Timer **callstackt, int stackidxt) 
    {
      using namespace gptl_util;
      int n;             // loop index through known parents
      Timer *pptr;       // pointer to parent in callstack
      Timer **pptrtmp;   // for realloc parent pointer array
      int nparent;       // number of parents
      int *parent_count; // number of times parent invoked this child
      static const char *thisfunc = "update_parent_info";

      if ( ! ptr )
	return -1;

      if (stackidxt < 0)
	return error ("%s: called with negative stackidx\n", thisfunc);

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
	  return error ("%s: realloc error pptrtmp nparent=%d\n", thisfunc, nparent);

	ptr->parent = pptrtmp;
	ptr->parent[nparent-1] = pptr;
	parent_count = (int *) realloc (ptr->parent_count, nparent * sizeof (int));
	if ( ! parent_count)
	  return error ("%s: realloc error parent_count nparent=%d\n", thisfunc, nparent);

	ptr->parent_count = parent_count;
	ptr->parent_count[nparent-1] = 1;
      }
      return 0;
    }

    /*
    ** preamble_stop: Do the things common to GPTLstop* routines
    **
    ** Input arguments:
    **   name: timer name
    **
    ** Output arguments:
    **   t: thread number
    **   tp1: time stamp
    **   usr: user time (if enabled)
    **   sys: system time (if enabled)
    **
    ** Return value: 0 (success) or GPTLerror (failure)
    */
    inline int preamble_stop (int *t, double *tp1, long *usr, long *sys, const char *name)
    {
      using namespace gptl_once;
      using namespace gptl_util;
      using namespace gptl_thread;
      static const char *thisfunc = "preamble_stop";
  
      if (disabled)
	return DONE;

      if ( ! initialized)
	return error ("%s timername=%s: GPTLinitialize has not been called\n", thisfunc, name);

      // Get the wallclock timestamp
      if (wallstats.enabled) {
	*tp1 = (*ptr2wtimefunc) ();
      }

      if (cpustats.enabled && get_cpustamp (usr, sys) < 0)
	return error ("%s: get_cpustamp error", name);

      if ((*t = get_thread_num ()) < 0)
	return error ("%s: bad return from get_thread_num\n", name);

      // If current depth exceeds a user-specified limit for print, just decrement and return
      if (stackidx[*t].val > depthlimit) {
	--stackidx[*t].val;
	return DONE;
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
    inline int update_stats (Timer *ptr, const double tp1, const long usr, const long sys,
				    const int t)
    {
      using namespace gptl_util;
      using namespace gptl_papi;
      double delta;      // difference
      int bidx;          // bottom of call stack
      Timer *bptr;       // pointer to last entry in call stack
      static const char *thisfunc = "update_stats";

      ptr->onflg = false;

#ifdef HAVE_PAPI
      if (dousepapi && PAPIstop (t, &ptr->aux) < 0)
	return error ("%s: error from PAPIstop\n", thisfunc);
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
	  gptl_util::warn ("%s: Imperfect nest detected: Got timer %s\n", thisfunc, name);
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
	return gptl_util::error ("%s: tree depth has become negative.\n", thisfunc);
      }
      return 0;
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
    inline unsigned int genhashidx (const char *name)
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
      indx = (MAX_CHARS*c[0] + (MAX_CHARS-mididx)*c[mididx] +
	      (MAX_CHARS-lastidx)*c[lastidx]) % tablesizem1 + 1;
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
    **   name:      name to be hashed
    **   indx:      hashtable index
    **
    ** Return value: pointer to the entry, or NULL if not found
    */
    inline Timer *getentry (const Hashentry *hashtable, const char *name, unsigned int indx)
    {
      Timer *ptr = 0;             // return value when entry not found

      // If nument exceeds 1 there was one or more hash collisions and we must search
      // linearly through the array of names with the same hash for a match
      for (int i = 0; i < hashtable[indx].nument; i++) {
	if (STRMATCH (name, hashtable[indx].entries[i]->name)) {
	  ptr = hashtable[indx].entries[i];
	  break;
	}
      }
      return ptr;
    }

    void check_memusage (const char *str, const char *funcnam)
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

#ifdef HAVE_NANOTIME
    // Copied from PAPI library
    inline long long nanotime (void)
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

    inline double utr_nanotime ()
    {
      return (double) (nanotime() * gptl_once::cyc2sec);
    }
#endif

#ifdef HAVE_LIBMPI
    inline double utr_mpiwtime ()
    {
      return MPI_Wtime();
    }
#endif

#ifdef HAVE_LIBRT
#include <time.h>
    inline double utr_clock_gettime ()
    {
      struct timespec tp;
      (void) clock_gettime (CLOCK_REALTIME, &tp);
      return (tp.tv_sec - gptl_once::ref_clock_gettime) + 1.e-9*tp.tv_nsec;
    }
#endif

#ifdef _AIX
    inline double utr_read_real_time ()
    {
      timebasestruct_t ibmtime;
      (void) read_real_time (&ibmtime, TIMEBASE_SZ);
      (void) time_base_to_time (&ibmtime, TIMEBASE_SZ);
      return (ibmtime.tb_high - ref_read_real_time) + 1.e-9*ibmtime.tb_low;
    }
#endif

#ifdef HAVE_GETTIMEOFDAY
#include <sys/time.h>
    inline double utr_gettimeofday ()
    {
      struct timeval tp;
      (void) gettimeofday (&tp, 0);
      return (tp.tv_sec - gptl_once::ref_gettimeofday) + 1.e-6*tp.tv_usec;
    }
#endif

    inline double utr_placebo ()
    {
      static const double zero = 0.;
      return zero;
    }

    // Use anonymous namespace for functions private to namespace gptl_private
    namespace {
      void print_callstack (int t, const char *caller)
      {
	printf ("Current callstack from %s:\n", caller);
	for (int idx = stackidx[t].val; idx > 0; --idx) {
	  printf ("%s\n", callstack[t][idx]->name);
	}
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
      inline int get_cpustamp (long *usr, long *sys)
      {
	using namespace gptl_util;
#ifdef HAVE_TIMES
	struct tms buf;

	(void) times (&buf);
	*usr = buf.tms_utime;
	*sys = buf.tms_stime;
	return 0;
#else
	return error ("GPTL: get_cpustamp: times() not available\n");
#endif
      }

      // set_fp_procsiz: Change file pointer from stderr to point to "procsiz.<rank>" once
      // MPI has been initialized
      inline void set_fp_procsiz ()
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
    }
  }
}
