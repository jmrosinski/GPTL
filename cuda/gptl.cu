/*
** gptl.c
** Author: Jim Rosinski
**
** Main file contains most user-accessible GPTL functions
*/

#include <stdlib.h>        /* malloc */
#include <stdio.h>
#include <string.h>        /* memcpy */

#include "./private.h"
#include "./gptl.h"

__device__ static Timer **timers = 0;             /* linked list of timers */
__device__ static Timer **last = 0;               /* last element in list */
__device__ static int *max_depth;                 /* maximum indentation level encountered */
__device__ static int *max_name_len;              /* max length of timer name */
__device__ static int nthreads = -1;                /* num threads. Init to bad value */
__device__ static int maxthreads = -1;              /* max threads */
__device__ static bool disabled = false;          /* Timers disabled? */
__device__ static bool initialized = false;       /* GPTLinitialize has been called */
__device__ static bool verbose = false;           /* output verbosity */
__device__ static bool imperfect_nest;                 /* e.g. start(A),start(B),stop(A) */

/* Options, print strings, and default enable flags */
__device__ static Hashentry **hashtable;    /* table of entries */
__device__ static Timer ***callstack;       /* call stack */
__device__ static int *stackidx;            /* index into callstack: */
__device__ static int tablesize;
__device__ static int tablesizem1;
__device__ static int maxtimers;            /* max number of timers to pass back to CPU */

extern "C" {

/* Local function prototypes */
__device__ static inline int get_warp_num (void);         /* get 0-based thread number */
__device__ static inline unsigned int genhashidx (const char *);
__device__ static inline Timer *getentry (const Hashentry *, const char *, unsigned int);
__device__ static inline int update_parent_info (Timer *, Timer **, int);
__device__ static inline int update_stats (Timer *, const long long, const int);
__device__ static int update_ll_hash (Timer *, int, unsigned int);
__device__ static inline int update_ptr (Timer *, const int);
__device__ static inline int my_strlen (const char *);
__device__ static inline char *my_strcpy (char *, const char *);
__device__ static inline int my_strcmp (const char *, const char *);

/* These are invoked only from gptl.c */
__device__ extern int GPTLinitialize_gpu (const int, const int, const int, const int);
__device__ extern int GPTLenable_gpu (void);
__device__ extern int GPTLdisable_gpu (void);
__device__ extern int GPTLreset_gpu (void);

/* VERBOSE is a debugging ifdef local to the rest of this file */
#undef VERBOSE

/*
** GPTLinitialize_gpu (): Initialization routine must be called from single-threaded
**   region before any other timing routines may be called.  The need for this
**   routine could be eliminated if not targetting timing library for threaded
**   capability. 
** return value: 0 (success) or GPTLerror (failure)
*/
__device__ int GPTLinitialize_gpu (const int verbose_in,
				   const int tablesize_in,
				   const int maxthreads_in,
				   const int maxtimers_in)
{
  int i;                 /* loop index */
  int w;                 /* warp index */
  long long t1, t2;      /* returned from underlying timer */
  static const char *thisfunc = "GPTLinitialize_gpu";

  if (initialized)
    printf ("%s: has already been called\n", thisfunc);
  

  // Set global vars from input args
  verbose     = verbose_in;
  maxthreads    = maxthreads_in;
  tablesize   = tablesize_in;
  tablesizem1 = tablesize_in - 1;
  maxtimers   = maxtimers_in;

  /* Allocate space for global arrays */
  printf ("Calling GPTLallocate (gpu) for callstack\n");
  callstack     = (Timer ***)    GPTLallocate (maxthreads * sizeof (Timer **), thisfunc);
  printf ("Calling GPTLallocate (gpu) for stackidx\n");
  stackidx      = (int *)        GPTLallocate (maxthreads * sizeof (int), thisfunc);
  timers        = (Timer **)     GPTLallocate (maxthreads * sizeof (Timer *), thisfunc);
  last          = (Timer **)     GPTLallocate (maxthreads * sizeof (Timer *), thisfunc);
  max_depth     = (int *)        GPTLallocate (maxthreads * sizeof (int), thisfunc);
  max_name_len  = (int *)        GPTLallocate (maxthreads * sizeof (int), thisfunc);
  hashtable     = (Hashentry **) GPTLallocate (maxthreads * sizeof (Hashentry *), thisfunc);

  /* Initialize array values */
  for (w = 0; w < maxthreads; w++) {
    max_depth[w]    = -1;
    max_name_len[w] = 0;
    callstack[w] = (Timer **) GPTLallocate (MAX_STACK * sizeof (Timer *), thisfunc);
    hashtable[w] = (Hashentry *) GPTLallocate (tablesize * sizeof (Hashentry), thisfunc);
    for (i = 0; i < tablesize; i++) {
      hashtable[w][i].nument = 0;
    }

    /* Make a timer "GPTL_ROOT" to ensure no orphans, and to simplify printing. */
    timers[w] = (Timer *) GPTLallocate (sizeof (Timer), thisfunc);
    memset (timers[w], 0, sizeof (Timer));
    (void) my_strcpy (timers[w]->name, "GPTL_ROOT");
    timers[w]->onflg = true;
    last[w] = timers[w];

    stackidx[w] = 0;
    callstack[w][0] = timers[w];
    for (i = 1; i < MAX_STACK; i++)
      callstack[w][i] = 0;
  }

  if (verbose) {
    t1 = clock64 ();
    t2 = clock64 ();
    if (t1 > t2)
      printf ("%s: negative delta-t=%lld\n", thisfunc, t2-t1);

    printf ("Per call overhead est. t2-t1=%g should be near zero\n", t2-t1);
    printf ("Underlying wallclock timing routine is clock64\n");
  }

  imperfect_nest = false;
  initialized = true;
  return 0;
}

/*
** GPTLfinalize_gpu (): Finalization routine must be called from single-threaded
**   region. Free all malloc'd space
**
** return value: 0 (success) or GPTLerror (failure)
*/
__device__ int GPTLfinalize_gpu (void)
{
  int w;                /* warp index */
  Timer *ptr, *ptrnext; /* ll indices */
  static const char *thisfunc = "GPTLfinalize_gpu";

  if ( ! initialized)
    return GPTLerror_1s ("%s: initialization was not completed\n", thisfunc);

  for (w = 0; w < maxthreads; ++w) {
    free (hashtable[w]);
    hashtable[w] = NULL;
    free (callstack[w]);
    for (ptr = timers[w]; ptr; ptr = ptrnext) {
      ptrnext = ptr->next;
      if (ptr->nparent > 0) {
        free (ptr->parent);
        free (ptr->parent_count);
      }
      if (ptr->nchildren > 0)
        free (ptr->children);
      free (ptr);
    }
  }

  free (callstack);
  free (stackidx);
  free (timers);
  free (last);
  free (max_depth);
  free (max_name_len);
  free (hashtable);

  GPTLreset_errors ();

  /* Reset initial values */
  timers = 0;
  last = 0;
  max_depth = 0;
  max_name_len = 0;
  nthreads = -1;
  maxthreads = -1;
  disabled = false;
  initialized = false;
  verbose = false;

  return 0;
}

/*
** GPTLstart: start a timer
**
** Input arguments:
**   name: timer name
**
** Return value: 0 (success) or GPTLerror (failure)
*/
__device__ int GPTLstart_gpu (const char *name)               /* timer name */
{
  Timer *ptr;        /* linked list pointer */
  int w;             /* warp index (of this thread) */
  int numchars;      /* number of characters to copy */
  unsigned int indx; /* hash table index */
  static const char *thisfunc = "GPTLstart";

  if (disabled)
    return 0;

  if ( ! initialized)
    return GPTLerror_2s ("%s name=%s: GPTLinitialize has not been called\n", thisfunc, name);

  if ((w = get_warp_num ()) == -1)
    return GPTLerror_1s ("%s: bad return from get_warp_num\n", thisfunc);

  // Return if not thread 0 of the warp
  if (w < 0)
    return 0;

  /* ptr will point to the requested timer in the current list, or NULL if this is a new entry */
  indx = genhashidx (name);
  ptr = getentry (hashtable[w], name, indx);

  /* 
  ** Recursion => increment depth in recursion and return.  We need to return 
  ** because we don't want to restart the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */
  if (ptr && ptr->onflg) {
    ++ptr->recurselvl;
    return 0;
  }

  /*
  ** Increment stackidx[w] unconditionally. This is necessary to ensure the correct
  ** behavior when GPTLstop decrements stackidx[w] unconditionally.
  */
  if (++stackidx[w] > MAX_STACK-1)
    return GPTLerror_1s ("%s: stack too big\n", thisfunc);

  if ( ! ptr) { /* Add a new entry and initialize */
    ptr = (Timer *) GPTLallocate (sizeof (Timer), thisfunc);
    memset (ptr, 0, sizeof (Timer));

    numchars = MIN (my_strlen (name), MAX_CHARS);
    memcpy (ptr->name, name, numchars);
    ptr->name[numchars] = '\0';

    if (update_ll_hash (ptr, w, indx) != 0)
      return GPTLerror_1s ("%s: update_ll_hash error\n", thisfunc);
  }

  if (update_parent_info (ptr, callstack[w], stackidx[w]) != 0)
    return GPTLerror_1s ("%s: update_parent_info error\n", thisfunc);

  if (update_ptr (ptr, w) != 0)
    return GPTLerror_1s ("%s: update_ptr error\n", thisfunc);

  return (0);
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
__device__ int GPTLinit_handle_gpu (const char *name,     /* timer name */
				int *handle)          /* handle (output if input value is zero) */
{
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
** Return value: 0 (success) or GPTLerror (failure)
*/
__device__ int GPTLstart_handle_gpu (const char *name,  /* timer name */
				 int *handle)       /* handle (output if input value is zero) */
{
  Timer *ptr;        /* linked list pointer */
  int w;             /* warp index (of this thread) */
  int numchars;      /* number of characters to copy */
  static const char *thisfunc = "GPTLstart_handle";

  if (disabled)
    return 0;

  if ( ! initialized)
    return GPTLerror_2s ("%s name=%s: GPTLinitialize has not been called\n", thisfunc, name);

  if ((w = get_warp_num ()) == -1)
    return GPTLerror_1s ("%s: bad return from get_warp_num\n", thisfunc);

  // Return if not thread 0 of the warp
  if (w < 0)
    return 0;

  /*
  ** If handle is zero on input, generate the hash entry and return it to the user.
  ** Otherwise assume it's a previously generated hash index passed in by the user.
  ** Don't need a critical section here--worst case multiple threads will generate the
  ** same handle and store to the same memory location, and this will only happen once.
  */
  if (*handle == 0) {
    *handle = (int) genhashidx (name);
#ifdef VERBOSE
    printf ("%s: name=%s warp %d generated handle=%d\n", thisfunc, name, w, *handle);
#endif
  } else if ((unsigned int) *handle > tablesizem1) {
    return GPTLerror_1s2d ("%s: Bad input handle=%u exceeds tablesizem1=%d\n", 
			   thisfunc, *handle, tablesizem1);
  }

  ptr = getentry (hashtable[w], name, (unsigned int) *handle);
  
  /* 
  ** Recursion => increment depth in recursion and return.  We need to return 
  ** because we don't want to restart the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */
  if (ptr && ptr->onflg) {
    ++ptr->recurselvl;
    return 0;
  }

  /*
  ** Increment stackidx[w] unconditionally. This is necessary to ensure the correct
  ** behavior when GPTLstop decrements stackidx[w] unconditionally.
  */
  if (++stackidx[w] > MAX_STACK-1)
    return GPTLerror_1s ("%s: stack too big\n", thisfunc);

  if ( ! ptr) { /* Add a new entry and initialize */
    ptr = (Timer *) GPTLallocate (sizeof (Timer), thisfunc);
    memset (ptr, 0, sizeof (Timer));

    numchars = MIN (my_strlen (name), MAX_CHARS);
    memcpy (ptr->name, name, numchars);
    ptr->name[numchars] = '\0';

    if (update_ll_hash (ptr, w, (unsigned int) *handle) != 0)
      return GPTLerror_1s ("%s: update_ll_hash error\n", thisfunc);
  }

  if (update_parent_info (ptr, callstack[w], stackidx[w]) != 0)
    return GPTLerror_1s ("%s: update_parent_info error\n", thisfunc);

  if (update_ptr (ptr, w) != 0)
    return GPTLerror_1s ("%s: update_ptr error\n", thisfunc);

  return (0);
}

/*
** update_ll_hash: Update linked list and hash table.
**                 Called by all GPTLstart* routines when there is a new entry
**
** Input arguments:
**   ptr:  pointer to timer
**   w:    warp index
**   indx: hash index
**
** Return value: 0 (success) or GPTLerror (failure)
*/
__device__ static int update_ll_hash (Timer *ptr, int w, unsigned int indx)
{
  int nchars;      /* number of chars */
  int nument;      /* number of entries */
  static const char *thisfunc = "update_ll_hash";

  nchars = my_strlen (ptr->name);
  if (nchars > max_name_len[w])
    max_name_len[w] = nchars;

  last[w]->next = ptr;
  last[w] = ptr;
  if (hashtable[w][indx].nument > MAXENT-1)
    return GPTLerror_2s ("%s: %s has too many hash collisions\n", thisfunc, ptr->name);

  ++hashtable[w][indx].nument;
  nument = hashtable[w][indx].nument;
  hashtable[w][indx].entries[nument-1] = ptr;
  return 0;
}

/*
** update_ptr: Update timer contents. Called by GPTLstart and GPTLstart_handle
**
** Input arguments:
**   ptr:  pointer to timer
**   w:    warp index
**
** Return value: 0 (success) or GPTLerror (failure)
*/
__device__ static inline int update_ptr (Timer *ptr, const int w)
{
  long long tp2;    /* time stamp */

  ptr->onflg = true;
  tp2 = clock64 ();
  ptr->wall.last = tp2;
  return 0;
}

/*
** update_parent_info: update info about parent, and in the parent about this child
**                     Called by all GPTLstart* routines
**
** Arguments:
**   ptr:  pointer to timer
**   callstackt: callstack for this warp
**   stackidxt:  stack index for this warp
**
** Return value: 0 (success) or GPTLerror (failure)
*/
__device__ static inline int update_parent_info (Timer *ptr, 
						 Timer **callstackt, 
						 int stackidxt) 
{
  int n;             /* loop index through known parents */
  Timer *pptr;       /* pointer to parent in callstack */
  int nparent;       /* number of parents */
  static const char *thisfunc = "update_parent_info";

  if ( ! ptr )
    return -1;

  if (stackidxt < 0)
    return GPTLerror_1s ("%s: called with negative stackidx\n", thisfunc);

  callstackt[stackidxt] = ptr;

  /* Bump orphan count if the region has no parent (should never happen since "GPTL_ROOT" added) */
  if (stackidxt == 0) {
    ++ptr->norphan;
    return 0;
  }

  pptr = callstackt[stackidxt-1];

  /* If this parent occurred before, bump its count */
  for (n = 0; n < ptr->nparent; ++n) {
    if (ptr->parent[n] == pptr) {
      ++ptr->parent_count[n];
      break;
    }
  }

  /* If this is a new parent, update info */
  if (n == ptr->nparent) {
    if (ptr->nparent == MAXPARENT)
      return GPTLerror_2s1d ("%s: cannot add newparent=%s because it exceeds MAXPARENT=%d\n",
			     thisfunc, pptr->name, MAXPARENT);
    ++ptr->nparent;
    nparent = ptr->nparent;

    ptr->parent[nparent-1] = pptr;
    ptr->parent_count[nparent-1] = 1;
  }
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
__device__ int GPTLstop_gpu (const char *name)               /* timer name */
{
  long long tp1 = 0;         /* time stamp */
  Timer *ptr;                /* linked list pointer */
  int w;                     /* warp number for this process */
  unsigned int indx;         /* index into hash table */
  static const char *thisfunc = "GPTLstop";

  if (disabled)
    return 0;

  if ( ! initialized)
    return GPTLerror_1s ("%s: GPTLinitialize has not been called\n", thisfunc);

  /* Get the timestamp */
    
  tp1 = clock64 ();

  if ((w = get_warp_num ()) == -1)
    return GPTLerror_1s ("%s: bad return from get_warp_num\n", thisfunc);

  // Return if not thread 0 of the warp
  if (w < 0)
    return 0;

  indx = genhashidx (name);
  if (! (ptr = getentry (hashtable[w], name, indx)))
    return GPTLerror_1s1d1s ("%s warp %d: timer for %s had not been started.\n", thisfunc, w, name);

  if ( ! ptr->onflg )
    return GPTLerror_2s ("%s: timer %s was already off.\n", thisfunc, ptr->name);

  ++ptr->count;

  /* 
  ** Recursion => decrement depth in recursion and return.  We need to return
  ** because we don't want to stop the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */
  if (ptr->recurselvl > 0) {
    --ptr->recurselvl;
    return 0;
  }

  if (update_stats (ptr, tp1, w) != 0)
    return GPTLerror_1s ("%s: error from update_stats\n", thisfunc);

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
__device__ int GPTLstop_handle_gpu (const char *name,     /* timer name */
				const int *handle)    /* handle */
{
  long long tp1 = 0;         /* time stamp */
  Timer *ptr;                /* linked list pointer */
  int w;                     /* warp number for this process */
  unsigned int indx;
  static const char *thisfunc = "GPTLstop_handle";

  if (disabled)
    return 0;

  if ( ! initialized)
    return GPTLerror_1s ("%s: GPTLinitialize has not been called\n", thisfunc);

  /* Get the timestamp */
  tp1 = clock64 ();

  if ((w = get_warp_num ()) == -1)
    return GPTLerror_1s ("%s: bad return from get_warp_num\n", thisfunc);

  // Return if not thread 0 of the warp
  if (w < 0)
    return 0;

  indx = (unsigned int) *handle;
  if (indx == 0 || indx > tablesizem1)
    return GPTLerror_1s1d1s ("%s: bad input handle=%u for timer %s.\n", thisfunc, (int) indx, name);
  
  if ( ! (ptr = getentry (hashtable[w], name, indx)))
    return GPTLerror_1s1d1s ("%s: handle=%u has not been set for timer %s.\n", 
			     thisfunc, (int) indx, name);

  if ( ! ptr->onflg )
    return GPTLerror_2s ("%s: timer %s was already off.\n", thisfunc, ptr->name);

  ++ptr->count;

  /* 
  ** Recursion => decrement depth in recursion and return.  We need to return
  ** because we don't want to stop the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */
  if (ptr->recurselvl > 0) {
    --ptr->recurselvl;
    return 0;
  }

  if (update_stats (ptr, tp1, w) != 0)
    return GPTLerror_1s ("%s: error from update_stats\n", thisfunc);

  return 0;
}

/*
** update_stats: update stats inside ptr. Called by GPTLstop, GPTLstop_handle
**
** Input arguments:
**   ptr: pointer to timer
**   tp1: input time stamp
**   w: warp index
**
** Return value: 0 (success) or GPTLerror (failure)
*/
__device__ static inline int update_stats (Timer *ptr, 
					   const long long tp1, 
					   const int w)
{
  long long delta;   /* difference */
  int bidx;          /* bottom of call stack */
  Timer *bptr;       /* pointer to last entry in call stack */
  static const char *thisfunc = "update_stats";

  ptr->onflg = false;

  delta = tp1 - ptr->wall.last;
  ptr->wall.accum += delta;

  if (delta < 0)
    printf ("GPTL: %s: negative delta=%lld\n", thisfunc, delta);

  if (ptr->count == 1) {
    ptr->wall.max = delta;
    ptr->wall.min = delta;
  } else {
    if (delta > ptr->wall.max)
      ptr->wall.max = delta;
    if (delta < ptr->wall.min)
      ptr->wall.min = delta;
  }

  /* Verify that the timer being stopped is at the bottom of the call stack */
  bidx = stackidx[w];
  bptr = callstack[w][bidx];
  if (ptr != bptr) {
    imperfect_nest = true;
    printf ("%s: Got timer=%s expected btm of call stack=%s\n", 
	    thisfunc, ptr->name, bptr->name);
  }

  --stackidx[w];           /* Pop the callstack */
  if (stackidx[w] < -1) {
    stackidx[w] = -1;
    return GPTLerror_1s ("%s: tree depth has become negative.\n", thisfunc);
  }

  return 0;
}

/*
** GPTLenable_gpu: enable timers
**
** Return value: 0 (success)
*/
  __device__ int GPTLenable_gpu (void)
{
  disabled = false;
  return (0);
}

/*
** GPTLdisable_gpu: disable timers
**
** Return value: 0 (success)
*/
__device__ int GPTLdisable_gpu (void)
{
  disabled = true;
  return (0);
}

/*
** GPTLreset_gpu: reset all timers to 0
**
** Return value: 0 (success) or GPTLerror (failure)
*/
__device__ int GPTLreset_gpu (void)
{
  int w;             /* index over threads */
  Timer *ptr;        /* linked list index */
  static const char *thisfunc = "GPTLreset";

  if ( ! initialized)
    return GPTLerror_1s ("%s: GPTLinitialize has not been called\n", thisfunc);

  for (w = 0; w < nthreads; w++) {
    for (ptr = timers[w]; ptr; ptr = ptr->next) {
      ptr->onflg = false;
      ptr->count = 0;
      memset (&ptr->wall, 0, sizeof (ptr->wall));
    }
  }

  if (verbose)
    printf ("%s: accumulators for all GPU timers set to zero\n", thisfunc);

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
__device__ static inline unsigned int genhashidx (const char *name)
{
  const unsigned char *c;       /* pointer to elements of "name" */
  unsigned int indx;            /* return value of function */
#ifdef NEWWAY
  unsigned int mididx, lastidx; /* mid and final index of name */

  lastidx = my_strlen (name) - 1;
  mididx = lastidx / 2;
#else
  int i;                        /* iterator (OLDWAY only) */
#endif
  /* 
  ** Disallow a hash index of zero (by adding 1 at the end) since user input of an uninitialized 
  ** value, though an error, has a likelihood to be zero.
  */
#ifdef NEWWAY
  c = (unsigned char *) name;
  indx = (MAX_CHARS*c[0] + (MAX_CHARS-mididx)*c[mididx] + (MAX_CHARS-lastidx)*c[lastidx]) % tablesizem1 + 1;
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
**   indx:      hashtable index
**
** Return value: pointer to the entry, or NULL if not found
*/
__device__ static inline Timer *getentry (const Hashentry *hashtable, /* hash table */
					  const char *name,           /* name to hash */
					  unsigned int indx)          /* hash index */
{
  int i;                      /* loop index */
  Timer *ptr = 0;             /* return value when entry not found */

  /* 
  ** If nument exceeds 1 there was one or more hash collisions and we must search
  ** linearly through the array of names with the same hash for a match
  */
  for (i = 0; i < hashtable[indx].nument; i++) {
    if (STRMATCH (name, hashtable[indx].entries[i]->name)) {
      ptr = hashtable[indx].entries[i];
      break;
    }
  }
  return ptr;
}

/*
** placebo: does nothing and returns zero always. Useful for estimating overhead costs
*/
__device__ static int init_placebo ()
{
  return 0;
}

__device__ static inline long long utr_placebo ()
{
  return (long long) 0;
}

__device__ static inline int get_warp_num ()
{
  static const char *thisfunc = "get_warp_num";
  int warpId;
  int blockId = blockIdx.x 
    + blockIdx.y * gridDim.x 
    + gridDim.x * gridDim.y * blockIdx.z; 
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
    + (threadIdx.z * (blockDim.x * blockDim.y))
    + (threadIdx.y * blockDim.x)
    + threadIdx.x;

#ifdef USE_WARP_NUM
  if (threadId % WARPSIZE != 0)
    return NOT_ROOT_OF_WARP;

  warpId = threadId / WARPSIZE;
#else
  warpId = threadId;
#endif
  if (warpId > maxthreads-1)
    return GPTLerror_1s2d ("%s: warpId=%d exceeds maxthreads=%d\n", thisfunc, warpId, maxthreads);
  else
    return warpId;
}

//JR want to use parameters to dimension arrays but nvcc is not C99 compliant
__device__ int GPTLfill_gpustats (Gpustats gpustats[][DEFAULT_MAXTIMERS_GPU], 
                                  int *max_name_len_out)
{
  int w;      // warp index
  int n;      // timer index
  int p;      // parent index
  Timer *ptr; // loop through linked list
  static const char *thisfunc = "GPTLfill_gpustats";

  for (w = 0; w < nthreads; ++w) {
    for (ptr = timers[w], n = 0; ptr; ptr = ptr->next, ++n) {
      if (n > maxtimers-1) 
	return GPTLerror_1s1d ("%s: Truncating timer output at %d entries."
			       "Suggest increase maxtimers array size to fill_output_stats()\n",
			       thisfunc, maxtimers);
      (void) my_strcpy (gpustats[w][n].name, ptr->name);
      gpustats[w][n].count       = ptr->count;
      gpustats[w][n].accum       = ptr->wall.accum;
      gpustats[w][n].max         = ptr->wall.max;
      gpustats[w][n].min         = ptr->wall.min;
      gpustats[w][n].nparent     = ptr->nparent;
      for (p = 0; p < ptr->nparent; ++p) {
	if (p > MAXPARENT-1) {
	  printf ("%s w=%d Timer=%s: Truncating parent count at MAXPARENT=%d out of %d\n", 
		  thisfunc, w, ptr->name, MAXPARENT, ptr->nparent);
	  break;
	}
	(void) my_strcpy (gpustats[w][n].parentname[p], ptr->parent[p]->name);
      }
    }
    max_name_len_out[w] = max_name_len[w];
  }
  return 0;
}

__device__ static inline int my_strlen (const char *str)
{
  int i;

  for (i = 0; str[i] != '\0'; ++i);
  return i;
}

__device__ static inline char *my_strcpy (char *dest, const char *src)
{
  char *ret = dest;

  while (*src != '\0')
    *dest++ = *src++;
  *dest = '\0';
  return ret;
}

__device__ static inline int my_strcmp (const char *str1, const char *str2)
{
  while (*str1 == *str2) {
    if (*str1 == '\0')
      break;
    ++str1;
    ++str2;
  }
  return (int) (*str1 - *str2);
}

}
