/*
$Id: private.h,v 1.30 2007-01-01 03:44:02 rosinski Exp $
*/

#include <stdio.h>
#include <sys/time.h>
#include "gptl.h"

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

#ifndef MAX
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#endif

#define STRMATCH(X,Y) (strcmp((X),(Y)) == 0)

/* longest timer name allowed (probably safe to just change) */
#define MAX_CHARS 31

/* max allowable number of PAPI counters (though most machines allow fewer */
#define MAX_AUX 8

#ifndef __cplusplus
typedef enum {false = 0, true = 1} bool;  /* mimic C++ */
#endif

typedef struct {
  long last_utime;          /* saved usr time from "start" */
  long last_stime;          /* saved sys time from "start" */
  long accum_utime;         /* accumulator for usr time */
  long accum_stime;         /* accumulator for sys time */
} Cpustats;

typedef struct {
  double last;              /* timestamp from last call */
  double accum;             /* accumulated time */
  float max;                /* longest time for start/stop pair */
  float min;                /* shortest time for start/stop pair */
} Wallstats;

typedef struct {
  long long last[MAX_AUX];  /* array of saved counters from "start" */
  long long accum[MAX_AUX]; /* accumulator for counters */
  long long accum_cycles;   /* for overhead computation */
} Papistats;
  
typedef struct TIMER {
  char name[MAX_CHARS+1];   /* timer name (user input) */
  bool onflg;               /* timer currently on or off */
  unsigned int depth;       /* depth in "calling" tree */
  unsigned int recurselvl;  /* recursion level */
  unsigned int max_recurse; /* max recursion level */
  unsigned long count;      /* number of start/stop calls */
  unsigned long nrecurse;   /* number of recursive start/stop calls */
  Wallstats wall;           /* wallclock stats */
  Cpustats cpu;             /* cpu stats */
#ifdef HAVE_PAPI
  Papistats aux;            /* PAPI stats  */
#endif 
  struct TIMER *next;       /* next timer in linked list */
} Timer;

typedef struct {
  unsigned int nument;      /* number of entries hashed to the same value */
  Timer **entries;          /* array of timers hashed to the same value */
} Hashentry;

/* Function prototypes */

extern int GPTLerror (const char *, ...);      /* print error msg and return */
extern void GPTLset_abort_on_error (bool val); /* set flag to abort on error */
extern void *GPTLallocate (const int);         /* malloc wrapper */
extern int threadinit (int *, int *);          /* initialize threading environment */
extern void threadfinalize (void);             /* finalize threading environment */
#if ( defined THREADED_PTHREADS )
extern int get_thread_num (int *, int *);      /* determine thread number */
#endif

#ifdef HAVE_PAPI
extern int GPTL_PAPIsetoption (const int, const int);
extern int GPTL_PAPIinitialize (const int);
extern int GPTL_PAPIstart (const int, Papistats *);
extern int GPTL_PAPIstop (const int, Papistats *);
extern void GPTL_PAPIprstr (FILE *, const bool);
extern void GPTL_PAPIpr (FILE *, const Papistats *, const int, const int, const bool);
extern void GPTL_PAPIadd (Papistats *, const Papistats *);
extern void GPTL_PAPIfinalize (int);
#endif
