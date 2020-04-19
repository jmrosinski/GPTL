/*
** $Id: private.h,v 1.74 2011-03-28 20:55:19 rosinski Exp $
**
** Author: Jim Rosinski
**
** Contains definitions private to GPTL and inaccessible to invoking user environment
*/

#ifndef _GPTL_PRIVATE_
#define _GPTL_PRIVATE_

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
#define MAX_CHARS 63

// Longest allowed symbol name for libunwind
#define MAX_SYMBOL_NAME 255

// A non-zero non-error flag for start/stop
#define DONE 1

/* 
** max allowable number of PAPI counters, or derived events. For convenience,
** set to max (# derived events, # papi counters required) so "avail" lists
** all available options.
*/
#define MAX_AUX 9

typedef enum {false = 0, true = 1} bool;  // mimic C++

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
  
typedef struct {
  int counter;      // PAPI or Derived counter
  char namestr[13]; // PAPI or Derived counter as string
  char str8[9];     // print string for output timers (8 chars)
} Entry;

typedef struct {
  Entry event;
  int numidx;       // derived event: PAPI counter array index for numerator
  int denomidx;     // derived event: PAPI counter array index for denominator
} Pr_event;

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
  unsigned int recurselvl;  // recursion level
  unsigned int nchildren;   // number of children
  unsigned int nparent;     // number of parents
  unsigned int norphan;     // number of times this timer was an orphan
  bool onflg;               // timer currently on or off
  char name[MAX_CHARS+1];   // timer name (user input)
  char *longname;           // For autoprofiled names, full name for diagnostic printing
} Timer;

typedef struct {
  Timer **entries;          // array of timers hashed to the same value
  unsigned int nument;      // number of entries hashed to the same value
} Hashentry;

// Function prototypes
extern int GPTLerror (const char *, ...);                  // print error msg and return
extern void GPTLwarn (const char *, ...);                  // print warning msg and return
extern void GPTLnote (const char *, ...);                  // print warning msg and return
extern void GPTLset_abort_on_error (bool val);             // set flag to abort on error
extern void GPTLreset_errors (void);                       // num_errors to zero
extern void *GPTLallocate (const int, const char *);       // malloc wrapper

extern int GPTLstart_instr (void *);                       // auto-instrumented start
extern int GPTLstop_instr (void *);                        // auto-instrumented stop
extern int GPTLis_initialized (void);                      // needed by MPI_Init wrapper
extern int GPTLget_overhead (FILE *,                       // file descriptor
			     double (*)(),                 // UTR()
			     Timer *(const Hashentry *, const char *, unsigned int), // getentry()
			     unsigned int (const char *),  // genhashidx()
			     Nofalse *,                    // stackidx
			     Timer ***,                    // callstack
			     const Hashentry *,            // hashtable
			     const int,                    // tablesize
			     bool,                         // dousepapi
			     int,                          // imperfect_nest
			     double *,                     // self_ohd
			     double *);                    // parent_ohd
extern void GPTLprint_hashstats (FILE *, int, Hashentry **, int);
extern void GPTLprint_memstats (FILE *, Timer **, int);
extern Timer **GPTLget_timersaddr (void);
// For now this one is local to gptl.c but that may change if needs calling from pr_summary
extern int GPTLrename_duplicate_addresses (void);

extern void __cyg_profile_func_enter (void *, void *);
extern void __cyg_profile_func_exit (void *, void *);

extern bool GPTLonlypr_rank0;     // flag says ignore all stdout/stderr print from non-zero ranks

#ifdef HAVE_PAPI
extern Entry GPTLeventlist[];     // list of PAPI-based events to be counted
extern int GPTLnevents;           // number of PAPI events (init to 0)

extern int GPTL_PAPIsetoption (const int, const int);
extern int GPTL_PAPIinitialize (const int, const bool, int *, Entry *);
extern int GPTL_PAPIstart (const int, Papistats *);
extern int GPTL_PAPIstop (const int, Papistats *);
extern void GPTL_PAPIprstr (FILE *);
extern void GPTL_PAPIpr (FILE *, const Papistats *, const int, const int, const double);
extern void GPTL_PAPIadd (Papistats *, const Papistats *);
extern void GPTL_PAPIfinalize (int);
extern void GPTL_PAPIquery (const Papistats *, long long *, int);
extern int GPTL_PAPIget_eventvalue (const char *, const Papistats *, double *);
extern bool GPTL_PAPIis_multiplexed (void);
extern void GPTL_PAPIprintenabled (FILE *);
extern void read_counters1000 (void);
extern int GPTLget_npapievents (void);
extern int GPTLcreate_and_start_events (const int);
#endif

#ifdef ENABLE_PMPI
extern Timer *GPTLgetentry (const char *);
extern int GPTLpmpi_setoption (const int, const int);
#endif

#endif // _GPTL_PRIVATE_
