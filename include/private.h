#ifndef PRIVATE_H
#define PRIVATE_H

#include "gptl.h"  // GPTL_Option
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Things visible only to GPTL namespaces and functions

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

#ifndef MAX
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#endif

#define STRMATCH(X,Y) (strcmp((X),(Y)) == 0)

// Max size of hash table needed due to handle split into hash+collision index which must
// fit in a 32-bit int
#define MAX_TABLESIZE 100000
#define MAX_NUMENT      1000

// Default size of hash table
#define DEFAULT_TABLE_SIZE 1023

// Output counts less than PRTHRESH will be printed as integers
#define PRTHRESH 1000000L

// Maximum allowed callstack depth
#define MAX_STACK 128

// longest timer name allowed (probably safe to just change)
#define MAX_CHARS 63

// Longest allowed symbol name for libunwind
#define MAX_SYMBOL_NAME 255

/* 
** max allowable number of PAPI counters, or derived events. For convenience,
** set to max (# derived events, # papi counters required) so "avail" lists
** all available options.
*/
#define MAX_AUX 9

namespace gptl_private {
  typedef struct {
    const GPTL_Option option;  // wall, cpu, etc.
    const char *str;           // descriptive string for printing
    bool enabled;              // flag
  } Settings;
  
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
  
  class Timer {
  public:
#ifdef ENABLE_PMPI
    double nbytes;            // number of bytes for MPI call
#endif
#ifdef HAVE_PAPI
    Papistats aux;            // PAPI stats 
#endif 
    Cpustats cpu;             // cpu stats
    Wallstats wall;           // wallclock stats
    long count;               // number of start/stop calls
    long nrecurse;            // number of recursive start/stop calls
    void *address;            // address of timer: used only by _instr routines
    Timer *next;              // next timer in linked list
    Timer **parent;           // array of parents
    Timer **children;         // array of children
    int *parent_count;        // array of call counts, one for each parent
    int recurselvl;           // recursion level
    int nchildren;            // number of children
    int nparent;              // number of parents
    int norphan;              // number of times this timer was an orphan
    bool onflg;               // timer currently on or off
    char name[MAX_CHARS+1];   // timer name (user input)
    char *longname;           // For autoprofiled names, full name for diagnostic printing
    Timer (const char *, void *);
  };

  
  typedef struct {
    Timer **entries;          // array of timers hashed to the same value
    int nument;               // number of entries hashed to the same value
  } Hashentry;
  
  typedef struct {
    int val;                  // depth in calling tree
    int padding[31];          // padding is to mitigate false cache sharing
  } Nofalse; 
  
  extern bool disabled;
  extern bool dousepapi;
  extern char unknown[];
  extern Timer **timers;
  extern Timer **last;
  extern Settings cpustats;
  extern Settings wallstats;
  extern Settings overheadstats;  
  extern Hashentry **hashtable;  // table of entries
  extern Timer ***callstack;
  extern Nofalse *stackidx;
  extern int tablesize;
  extern int tablesizem1;
  extern float rssmax;
  extern bool imperfect_nest;
  extern FILE *fp_procsiz;

  // Anonymous namespace for local function prototypes
  namespace {
    extern "C" {
#ifdef HAVE_NANOTIME
      inline long long nanotime (void); // read counter (assembler)
#endif
      void print_callstack (int, const char *);
      inline int get_cpustamp (long *, long *);
      inline int get_cpustamp (long *, long *);
      inline void set_fp_procsiz (void);
    }
  }
  
  // Function prototypes visible only to GPTL routines
  extern "C" {
    extern double (*ptr2wtimefunc)(void);    // The underlying timing routine
    void check_memusage (const char *, const char *);
    inline int genhashidx (const char *);
    inline Timer *getentry (const Hashentry *, const char *, int);
    inline Timer *getentry_handle (const Hashentry *, const char *, int *, bool *);
    inline int preamble_start (int *, const char *);
    inline int update_parent_info (Timer *, Timer **, int);
    inline int preamble_stop (int *, double *, long *, long *, const char *);
    inline int update_stats (Timer *, const double, const long, const long, const int);
    int update_ll_hash (Timer *, int, const int);
    inline int update_ptr (Timer *, const int);
    // These are the (possibly) supported underlying wallclock timers
#ifdef HAVE_NANOTIME
    inline double utr_nanotime (void);
#endif 
#ifdef HAVE_LIBMPI
    inline double utr_mpiwtime (void);
#endif
#ifdef _AIX
    inline double utr_read_real_time (void);
#endif
#ifdef HAVE_LIBRT
    inline double utr_clock_gettime (void);
#endif
#ifdef HAVE_GETTIMEOFDAY
    inline double utr_gettimeofday (void);
#endif
    inline double utr_placebo (void);
  }
}
#endif
