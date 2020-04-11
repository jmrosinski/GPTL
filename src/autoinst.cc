#include "config.h" // Must be first include.
#include "private.h"
#include "autoinst.h"
#include "once.h"
#include "util.h"

#include <string.h>    // memset, strlen
#include <stdlib.h>    // malloc, free

#ifdef HAVE_LIBUNWIND
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#endif

#ifdef HAVE_BACKTRACE
#include <execinfo.h>
#endif

namespace gptl_autoinst {
  // prototypes for functions in anonymous namespace
  namespace {
    extern "C" {
      int get_symnam (void *, char **);
      void extract_name (char *, char **, void *);
#if ( defined HAVE_LIBUNWIND || defined HAVE_BACKTRACE )
      void __cyg_profile_func_enter (void *, void *);
      void __cyg_profile_func_exit (void *, void *);
#endif
    }
  }

  extern "C" {
    // APPEND_ADDRESS is an autoprofiling debugging ifdef that will append function address to name
#undef APPEND_ADDRESS

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
    inline Timer *getentry_instr (const Hashentry *hashtable, void *self, unsigned int *indx)
    {
      using namespace gptl_private;
      Timer *ptr = NULL;  // init to return value when entry not found

      /*
      ** Hash index is timer address modulo the table size
      ** On most machines, right-shifting the address helps because linkers often
      ** align functions on even boundaries
      */
      *indx = (((unsigned long) self) >> 4) % tablesize;
      for (int i = 0; i < hashtable[*indx].nument; ++i) {
	if (hashtable[*indx].entries[i].address == self) {
	  ptr = hashtable[*indx].entries[i].timer;
	  break;
	}
      }
      return ptr;
    }

#if ( defined HAVE_LIBUNWIND || defined HAVE_BACKTRACE )
    void __cyg_profile_func_enter (void *this_fn, void *call_site)
    {
      using namespace gptl_once;
      using namespace gptl_private;
      using namespace gptl_util;
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
    
      // Call preamble_start rather than just get_thread_num because preamble_stop is needed for
      // other reasons in __cyg_profile_func_exit, and the preamble* functions need to mirror each
      // other.
  
      if (gptl_private::preamble_start (&t, unknown) != 0)
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
	gptl_util::warn ("%s: stack too big\n", thisfunc);
	return;
      }

      if ( ! ptr) {     // Add a new entry and initialize
	if (get_symnam (this_fn, &symnam) != 0) {
	  printf ("%s: failed to find symbol for address %p\n", thisfunc, this_fn);
	  return;
	}
	ptr = new Timer (symnam, this_fn);
	free (symnam);
      
	if (update_ll_hash (ptr, t, indx) != 0) {
	  gptl_util::warn ("%s: update_ll_hash error\n", thisfunc);
	  return;
	}
      }
    
      if (update_parent_info (ptr, callstack[t], stackidx[t].val) != 0) {
	gptl_util::warn ("%s: update_parent_info error\n", thisfunc);
	return;
      }

      if (update_ptr (ptr, t) != 0) {
	gptl_util::warn ("%s: update_ptr error\n", thisfunc);
	return;
      }

      if (dopr_memusage && t == 0)
	check_memusage ("Begin", ptr->name);
    }

    // Use anonymous namespace for functions private to the outer namespace
    namespace {
#ifdef HAVE_BACKTRACE
      int get_symnam (void *this_fn, char **symnam)
      {
	using namespace gptl_util;
	char **strings = 0;
	void *buffer[3];
	int nptrs;
	char addrstr[MAX_CHARS+1];          // function address as a string
	static const char *thisfunc = "get_symnam(backtrace)";

	nptrs = backtrace (buffer, 3);
	if (nptrs != 3) {
	  warn ("%s backtrace failed nptrs should be 2 but is %d\n", thisfunc, nptrs);
	  return -1;
	}

	if ( ! (strings = backtrace_symbols (buffer, nptrs))) {
	  warn ("%s backtrace_symbols failed strings is null\n", thisfunc);
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
      int get_symnam (void *this_fn, char **symnam)
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
	    gptl_util::warn ("%s: unw_step failed\n", thisfunc);
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
	using namespace gptl_once;
	using namespace gptl_private;
	using namespace gptl_util;
	float rss;
	int t;             // thread index
	unsigned int indx; // hash table index
	Timer *ptr;        // pointer to entry if it already exists
	double tp1 = 0.0;  // time stamp */
	long usr = 0;      // user time (returned from get_cpustamp)
	long sys = 0;      // system time (returned from get_cpustamp)
	static const char *thisfunc = "__cyg_profile_func_exit";

	if (gptl_private::preamble_stop (&t, &tp1, &usr, &sys, thisfunc) != 0)
	  return;
       
	ptr = getentry_instr (hashtable[t], this_fn, &indx);

	if ( ! ptr) {
	  warn ("%s: timer for %p had not been started.\n", thisfunc, this_fn);
	  return;
	}

	if ( ! ptr->onflg ) {
	  warn ("%s: timer %s was already off.\n", thisfunc, ptr->name);
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

	if (gptl_private::update_stats (ptr, tp1, usr, sys, t) != 0) {
	  gptl_util::warn ("%s: error from update_stats\n", thisfunc);
	  return;
	}

	if (dopr_memusage && t == 0)
	  check_memusage ("End", ptr->name);
      }
#endif // HAVE_LIBUNWIND || HAVE_BACKTRACE
      // auto-instrument function for xlc: -qdebug=function_trace
#ifdef _AIX
#include <sys/systemcfg.h>
      static time_t ref_read_real_time = -1; // ref start point for read_real_time
      
      void __func_trace_enter (const char *function_name, const char *file_name, int line_number,
			       void **const user_data)
      {
	using namespace gptl_once;
	using namespace gptl_private;
	if (dopr_memusage && get_thread_num() == 0)
	  check_memusage ("Begin", function_name);
	(void) GPTLstart (function_name);
      }
  
      void __func_trace_exit (const char *function_name, const char *file_name, int line_number,
			      void **const user_data)
      {
	using namespace gptl_once;
	using namespace gptl_private;
	(void) GPTLstop (function_name);
	if (dopr_memusage && get_thread_num() == 0)
	  check_memusage ("End", function_name);
      }
#endif
    }
  }
}

