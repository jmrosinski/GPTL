/*
$Id: private.h,v 1.44 2008-05-11 02:59:57 rosinski Exp $
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

/* Maximum allowed stack depth */
#define MAX_STACK 128

/* longest timer name allowed (probably safe to just change) */
#define MAX_CHARS 31

/* max allowable number of PAPI counters (though most machines allow fewer */
#define MAX_AUX 16

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
  
typedef struct {
  int counter;      /* PAPI counter */
  char *counterstr; /* PAPI counter as string */
  char *prstr;      /* print string for output timers (16 chars) */
  char *str;        /* descriptive print string (more descriptive than prstr) */
} Papientry;

typedef struct TIMER {
  char name[MAX_CHARS+1];   /* timer name (user input) */
#ifdef HAVE_PAPI
  Papistats aux;            /* PAPI stats  */
#endif 
  Wallstats wall;           /* wallclock stats */
  Cpustats cpu;             /* cpu stats */
  unsigned long count;      /* number of start/stop calls */
  unsigned long nrecurse;   /* number of recursive start/stop calls */
  void *address;            /* address of timer: used only by _instr routines */
  struct TIMER *next;       /* next timer in linked list */
  struct TIMER **parent;    /* array of parents */
  struct TIMER **children;  /* array of children */
  int *parent_count;        /* array of call counts, one for each parent */
  unsigned int depth;       /* depth in "calling" tree */
  unsigned int recurselvl;  /* recursion level */
  unsigned int max_recurse; /* max recursion level */
  unsigned int nchildren;   /* number of children */
  unsigned int nparent;     /* number of parents */
  unsigned int norphan;     /* number of times this timer was an orphan */
  bool onflg;               /* timer currently on or off */
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
extern int GPTL_PAPIinitialize (const int, const bool, int *, Papientry *);
extern int GPTL_PAPIstart (const int, Papistats *);
extern int GPTL_PAPIstop (const int, Papistats *);
extern void GPTL_PAPIprstr (FILE *, const bool);
extern void GPTL_PAPIpr (FILE *, const Papistats *, const int, const int, const double, const bool);
extern void GPTL_PAPIadd (Papistats *, const Papistats *);
extern void GPTL_PAPIfinalize (int);
extern void GPTL_PAPIquery (const Papistats *, long long *, int);
extern bool GPTL_PAPIis_multiplexed (void);
extern void GPTL_PAPIprintenabled (FILE *);
#endif
