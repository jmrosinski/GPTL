/*
** $Id: private.h,v 1.74 2011-03-28 20:55:19 rosinski Exp $
**
** Author: Jim Rosinski
**
** Contains definitions private to GPTL and inaccessible to invoking user environment
*/

#ifndef _GPTL_PRIVATE_
#define _GPTL_PRIVATE_

#include "gptl.h"      // GPTLoption
#include <stdio.h>
#include <sys/time.h>

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

#ifndef MAX
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#endif

#define STRMATCH(X,Y) (strcmp((X),(Y)) == 0)

// Output counts less than PRTHRESH will be printed as integers
#define PRTHRESH 1000000L

// Maximum allowed callstack depth
#define MAX_STACK 128

// longest timer name allowed (probably safe to just change)
// Must be at least 16 to hold auto-profiled name, and 9 to hold "GPTL_ROOT"
#define MAX_CHARS 63

// Longest allowed symbol name for libunwind
#define MAX_SYMBOL_NAME 255

// A non-zero non-error flag for start/stop
#define DONE 1

// max allowable number of PAPI counters, or derived events.
#define MAX_AUX 3

// Size of table containing entries. Too small means many collisions which impedes performance
// Too big means less chance for the table to remain cache-resident.
#define DEFAULT_TABLE_SIZE 1023

typedef struct {
  int val;                  // depth in calling tree
  int padding[31];          // padding is to mitigate false cache sharing
} Nofalse; 

typedef struct {
  long last_utime;          // saved usr time from "start"
  long last_stime;          // saved sys time from "start"
  long accum_utime;         // accumulator for usr time
  long accum_stime;         // accumulator for sys time
} Cpustats;

typedef struct {
  double last;              // timestamp from last call
  double latest;            // most recent delta
  double accum;             // accumulated time
  float max;                // longest time for start/stop pair
  float min;                // shortest time for start/stop pair
} Wallstats;

typedef struct {
  long long last[MAX_AUX];  // array of saved counters from "start"
  long long accum[MAX_AUX]; // accumulator for counters
} Papistats;

typedef struct TIMER {
#ifdef ENABLE_PMPI
  double nbytes;            // number of bytes for MPI call
#endif
#ifdef HAVE_PAPI
  Papistats aux;            // PAPI stats 
#endif 
  Cpustats cpu;             // cpu stats
  Wallstats wall;           // wallclock stats
  unsigned long count;      // number of start/stop calls
  unsigned long nrecurse;   // number of recursive start/stop calls
#ifdef COLLIDE
  unsigned long collide;    // number of extra comparisons due to collision
#endif
  void *address;            // address of timer: used only by _instr routines
  struct TIMER *next;       // next timer in linked list
  struct TIMER **parent;    // array of parents
  struct TIMER **children;  // array of children
  int *parent_count;        // array of call counts, one for each parent
#ifdef ENABLE_NESTEDOMP
  int major;                // outer thread id in nest
  int minor;                // inner thread id in nest
#endif
  unsigned int recurselvl;  // recursion level
  unsigned int nchildren;   // number of children
  unsigned int nparent;     // number of parents
  unsigned int norphan;     // number of times this timer was an orphan
  int numchars;             // length of name
  bool onflg;               // timer currently on or off
  char name[MAX_CHARS+1];   // timer name (user input)
  char *longname;           // For autoprofiled names, full name for diagnostic printing
} Timer;
  
typedef struct {
  Timer **entries;          // array of timers hashed to the same value
  unsigned int nument;      // number of entries hashed to the same value
} Hashentry;

typedef struct {
  GPTLoption option;  // wall, cpu, etc.
  char *str;          // descriptive string for printing
  bool enabled;             // flag
} Settings;

// Function prototypes
extern "C" {
  int GPTLerror (const char *, ...);                  // print error msg and return
  void GPTLwarn (const char *, ...);                  // print warning msg and return
  void GPTLnote (const char *, ...);                  // print warning msg and return
  void GPTLset_abort_on_error (bool val);             // set flag to abort on error
  void GPTLreset_errors (void);                       // num_errors to zero
  void *GPTLallocate (const int, const char *);       // malloc wrapper

  void GPTLprint_memstats (FILE *, Timer **, int);
  Timer **GPTLget_timersaddr (void);
  // For now this one is local to gptl.c but that may change if needs calling from pr_summary
  int GPTLrename_duplicate_addresses (void);

  // Don't need these with C++
  // void __cyg_profile_func_enter (void *, void *);
  // void __cyg_profile_func_exit (void *, void *);

#ifdef ENABLE_PMPI
  Timer *GPTLgetentry (const char *);
  int GPTLpmpi_setoption (const int, const int);
#endif
}
#endif // _GPTL_PRIVATE_
