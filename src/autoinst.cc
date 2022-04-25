#include "config.h" // Must be first include.
#include "private.h"
#include "main.h"
#include "autoinst.h"
#include "memusage.h"
#include "thread.h"
#include "util.h"

#include <string.h>
#include <stdlib.h>

#ifdef HAVE_LIBUNWIND
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#endif

#ifdef HAVE_BACKTRACE
#include <execinfo.h>
static void extract_name (char *, char **, void *, const int);
#endif

static const char unknown[] = "unknown";

/*
** Add entry points for auto-instrumented codes
** Auto instrumentation flags for various compilers:
**
** gcc, pathcc, icc: -finstrument-functions
** pgcc:             -Minstrument:functions
** xlc:              -qdebug=function_trace
*/

#ifdef _AIX
extern "C" void __func_trace_enter (const char *function_name, const char *file_name, int line_number,
				    void **const user_data)
{
  if (dopr_memusage && thread::get_thread_num() == 0)
    check_memusage ("Begin", function_name);
  (void) GPTLstart (function_name);
}
  
extern "C" void __func_trace_exit (const char *function_name, const char *file_name, int line_number,
				   void **const user_data)
{
  (void) GPTLstop (function_name);
  if (dopr_memusage && thread::get_thread_num() == 0)
    check_memusage ("End", function_name);
}
  
#else
//_AIX not defined

#if ( defined HAVE_LIBUNWIND || defined HAVE_BACKTRACE )
extern "C" void __cyg_profile_func_enter (void *this_fn, void *call_site)
{
  int t;                // thread index
  int symsize;          // number of characters in symbol
  char *symnam = NULL;  // symbol name whether using unwind or backtrace
  int numchars;         // number of characters in function name
  unsigned int indx;    // hash table index
  Timer *ptr;           // pointer to entry if it already exists
  static const char *thisfunc = "__cyg_profile_func_enter";

  // In debug mode, get symbol name up front to diagnose function name
  // Otherwise live with "unknown" because getting the symbol name is very expensive

  // Call preamble_start rather than just get_thread_num because preamble_stop is needed for
  // other reasons in __cyg_profile_func_exit, and the preamble* functions need to mirror each
  // other.
  
  if (gptlmain::preamble_start (&t, unknown) != 0)
    return;
  
  ptr = autoinst::getentry_instr (gptlmain::hashtable[t], this_fn, &indx);

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
  if (++gptlmain::stackidx[t].val > MAX_STACK-1) {
    util::warn ("%s: stack too big\n", thisfunc);
    return;
  }

  // Nasty bit of code needs to be this way because separating into functions can cause
  // compilers to inline and screw things up
  if ( ! ptr) {     // Add a new entry and initialize
#if ( defined HAVE_BACKTRACE )
    char **strings = 0;
    void *buffer[2];
    int nptrs;
    
    nptrs = backtrace (buffer, 2);
    if (nptrs != 2) {
      util::warn ("%s backtrace failed nptrs should be 2 but is %d\n", thisfunc, nptrs);
      return;
    }

    if ( ! (strings = backtrace_symbols (buffer, nptrs))) {
      util::warn ("%s backtrace_symbols failed strings is null\n", thisfunc);
      return;
    }
    // extract_name will malloc space for symnam, and it will be freed below
    extract_name (strings[1], &symnam, this_fn, t);
    free (strings);  // backtrace() allocated strings

#elif ( defined HAVE_LIBUNWIND )

    char symbol[MAX_SYMBOL_NAME+1];
    unw_cursor_t cursor;
    unw_context_t context;
    unw_word_t offset, pc;
    // Initialize cursor to current frame for local unwinding.
    unw_getcontext (&context);
    unw_init_local (&cursor, &context);

    if (unw_step (&cursor) <= 0) { // unw_step failed: give up
      util::warn ("%s: unw_step failed\n", thisfunc);
      return;
    }

    unw_get_reg (&cursor, UNW_REG_IP, &pc);
    if (unw_get_proc_name (&cursor, symbol, sizeof(symbol), &offset) == 0) {
      int nchars = strlen (symbol);;
#ifdef APPEND_ADDRESS
      char addrname[16+2];
      symnam = (char *) malloc (nchars+16+2);  // 16 is nchars, +2 is for '#' and '\0'
      strncpy (symnam, symbol, nchars+1);
      snprintf (addrname, 16+2, "#%-16p", this_fn);
      strcat (symnam, addrname);
#else
      symnam = (char *) malloc (nchars + 1);
      strncpy (symnam, symbol, nchars+1);
#endif
    } else {
      // Name not found: write function address into symnam. Allow 16 characters to hold address
      symnam = (char *) malloc (16+1);
      snprintf (symnam, 16+1, "%-16p", this_fn);
    }
#endif

    // Whether backtrace or libunwind, symnam has now been defined
    symsize = strlen (symnam);
    ptr = (Timer *) util::allocate (sizeof (Timer), thisfunc);
    memset (ptr, 0, sizeof (Timer));
#ifdef ENABLE_NESTEDOMP
    thread::get_nested_thread_nums (&ptr->major, &ptr->minor);
#endif

    // Do the things specific to auto-profiled functions, then call update_ll_hash()
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

    if (gptlmain::update_ll_hash (ptr, t, indx) != 0) {
      util::warn ("%s: update_ll_hash error\n", thisfunc);
      return;
    }
  }

  if (gptlmain::update_parent_info (ptr, gptlmain::callstack[t], gptlmain::stackidx[t].val) != 0) {
    util::warn ("%s: update_parent_info error\n", thisfunc);
    return;
  }

  if (gptlmain::update_ptr (ptr, t) != 0) {
    util::warn ("%s: update_ptr error\n", thisfunc);
    return;
  }

  if (gptlmain::dopr_memusage && t == 0)
    memusage::check_memusage ("Begin", ptr->name);
}

#ifdef HAVE_BACKTRACE
// Backtrace strings have a bunch of extra stuff in them.
// Find the start and end of the function name, and return it in *symnam
// Note str gets modified by writing \0 after the end of the function name
static void extract_name (char *str, char **symnam, void *this_fn, const int t)
{
  int nchars;
  char *savetoken;
  char *saveptr[thread::nthreads];
  char *token;

#ifdef __APPLE__
  token = strtok_r (str, " ", &saveptr[t]);
  for (int n = 0; n < 3; ++n)
    token = strtok_r (NULL, " ", &saveptr[t]);
#else
  token = strtok_r (str, "(", &saveptr[t]);
  savetoken = token;
  if ( ! (token = strtok_r (NULL, "+", &saveptr[t])))
    token = savetoken;
#endif

  if ( ! token) {
    // Name not found: write function address into symnam. Allow 16 characters to hold address
    *symnam = (char *) malloc (16+1);
    snprintf (*symnam, 16+1,"%-16p", this_fn);
  } else {
    nchars = strlen (token);
#ifdef APPEND_ADDRESS
    char addrname[16+2];
    *symnam = (char *) malloc (nchars+16+2);  // 16 is nchars, +2 is for '#' and '\0'
    strncpy (*symnam, token, nchars+1);
    snprintf (addrname, 16+2, "#%-16p", this_fn);
    strcat (*symnam, addrname);
#else
    *symnam = (char *) malloc (nchars + 1);
    strncpy (*symnam, token, nchars+1);
#endif
  }
}
#endif   // HAVE_BACKTRACE

extern "C" void __cyg_profile_func_exit (void *this_fn, void *call_site)
{
  int t;                     // thread index
  unsigned int indx;         // hash table index
  Timer *ptr;                // pointer to entry if it already exists
  double tp1 = 0.0;          // time stamp
  long usr = 0;              // user time (returned from get_cpustamp)
  long sys = 0;              // system time (returned from get_cpustamp)
  static const char *thisfunc = "__cyg_profile_func_exit";

  if (gptlmain::preamble_stop (&t, &tp1, &usr, &sys, unknown) != 0)
    return;
       
  ptr = autoinst::getentry_instr (gptlmain::hashtable[t], this_fn, &indx);

  if ( ! ptr) {
    util::warn ("%s: timer for %p had not been started.\n", thisfunc, this_fn);
    return;
  }

  if ( ! ptr->onflg ) {
    util::warn ("%s: timer %s was already off.\n", thisfunc, ptr->name);
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

  if (gptlmain::update_stats (ptr, tp1, usr, sys, t) != 0) {
    util::warn ("%s: error from update_stats\n", thisfunc);
    return;
  }

  if (gptlmain::dopr_memusage && t == 0)
    memusage::check_memusage ("End", ptr->name);
}
#endif // HAVE_LIBUNWIND || HAVE_BACKTRACE
#endif // _AIX false branch

namespace autoinst {
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
  extern "C" Timer *getentry_instr (const Hashentry *hashtable, void *self, unsigned int *indx)
  {
    int i;
    Timer *ptr = NULL;  // init to return value when entry not found

    /*
    ** Hash index is timer address modulo the table size
    ** On most machines, right-shifting the address helps because linkers often
    ** align functions on even boundaries
    */
    *indx = (((unsigned long) self) >> 4) % gptlmain::tablesize;
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
}
