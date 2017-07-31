#include <stdio.h>
#include <string.h>
#include "private.h"

__device__ static void misc_sim (Nofalse *, Timer ***, int);
__device__ static bool initialized = true;
__device__ static bool disabled = false;

/*
** All routines in this file are non-public
*/

/*
** GPTLget_overhead: return current status info about a timer. If certain stats are not enabled, 
** they should just have zeros in them.
** 
** Input args:
**   getentry:      From gptl.c, finds the entry in the hash table
**   genhashidx:    From gptl.c, generates the hash index
**   get_thread_num:From gptl.c, gets the thread number
**   hashtable:     hashtable for thread 0
**   tablesize:     size of hashtable
**
** Output args:
**   get_thread_num_ohd: Getting my thread index
**   genhashidx_ohd:     Generating hash index
**   getentry_ohd:       Finding entry in hash table
**   utr_ohd:            Underlying timer routine
**   misc_ohd:           Misc. calcs within start/stop
**   self_ohd:           Estimate of GPTL-induced overhead in the timer itself (included in "Wallclock")
**   parent_ohd:         Estimate of GPTL-induced overhead for the timer which appears in its parents
*/
__device__ int GPTLget_overhead (Timer *getentry (const Hashentry *, const char *, unsigned int),
				 unsigned int genhashidx (const char *),
				 int get_thread_num (void),
				 int *stackidx,
				 Timer ***callstack,
				 const Hashentry *hashtable, 
				 const int tablesize,
				 int imperfect_nest,

				 long long get_thread_num_ohd; /* Getting my thread index */
				 long long genhashidx_ohd;     /* Generating hash index */
				 long long getentry_ohd;       /* Finding entry in hash table */
				 long long utr_ohd;            /* Underlying timing routine */
				 long long misc_ohd;           /* misc. calcs within start/stop */
				 long long self_ohd,
				 long long parent_ohd)
{
  long long t1, t2;          /* Initial, final timer values */
  int i, n;
  int ret;
  int mythread;              /* which thread are we */
  unsigned int hashidx;      /* Hash index */
  int randomvar;             /* placeholder for taking the address of a variable */
  Timer *entry;              /* placeholder for return from "getentry()" */
  static const char *thisfunc = "GPTLget_overhead";

  /*
  ** Gather timings by running kernels 1000 times each
  ** First: get_thread_num() overhead
  */
  t1 = clock64();
  for (i = 0; i < 1000; ++i) {
    mythread = get_thread_num ();
  }
  t2 = clock64();
  get_thread_num_ohd = (t2 - t1) / 1000;

  /* genhashidx overhead */
  t1 = clock64();
  for (i = 0; i < 1000; ++i) {
    hashidx = genhashidx ("timername");
  }
  t2 = clock64();
  genhashidx_ohd = (t2 - t1) / 1000;

  /* 
  ** getentry overhead
  ** Find the first hashtable entry with a valid name. Start at 1 because 0 is not a valid hash
  */
  for (n = 1; n < tablesize; ++n) {
    if (hashtable[n].nument > 0 && strlen (hashtable[n].entries[0]->name) > 0) {
      hashidx = genhashidx (hashtable[n].entries[0]->name);
      t1 = clock64();
      for (i = 0; i < 1000; ++i)
	entry = getentry (hashtable, hashtable[n].entries[0]->name, hashidx);
      t2 = clock64();
      break;
    }
  }
  if (n == tablesize) {
    t1 = clock64();
    for (i = 0; i < 1000; ++i)
      entry = getentry (hashtable, "timername", hashidx);
    t2 = clock64();
  }
  getentry_ohd = (t2 - t1) / 1000;
  /* utr overhead */
  t1 = clock64();
  for (i = 0; i < 1000; ++i) {
    t2 = clock64();
  }
  utr_ohd = (t2 - t1) / 1000;

  /* misc start/stop overhead */
  if (imperfect_nest) {
    misc_ohd = 0;
  } else {
    t1 = clock64();
    for (i = 0; i < 1000; ++i) {
      misc_sim (stackidx, callstack, 0);
    }
    t2 = clock64();
    misc_ohd = (t2 - t1) / 1000;
  }

  self_ohd   = utr_ohd;
  parent_ohd = utr_ohd + misc_ohd + 2*(get_thread_num_ohd + genhashidx_ohd + getentry_ohd);
  return 0;
}

/*
** misc_sim: Simulate the cost of miscellaneous computations in start/stop
** 
** Input args:
**   stackidx:  stack index
**   callstack: call stack
**   t:         thread index
*/
__device__ static void misc_sim (Nofalse *stackidx, Timer ***callstack, int t)
{
  int bidx;
  Timer *bptr;
  static Timer *ptr = 0;
  static const char *thisfunc = "misc_sim";

  if (disabled)
    printf ("GPTL: %s: should never print disabled\n", thisfunc);

  if (! initialized)
    printf ("GPTL: %s: should never print ! initialized\n", thisfunc);

  bidx = stackidx[t].val;
  bptr = callstack[t][bidx];
  if (ptr == bptr)
    printf ("GPTL: %s: should never print ptr=bptr\n", thisfunc);

  --stackidx[t].val;
  if (stackidx[t].val < -2)
    printf ("GPTL: %s: should never print stackidxt < -2\n", thisfunc);

  if (++stackidx[t].val > MAX_STACK-1)
    printf ("GPTL: %s: should never print stackidxt > MAX_STACK-1\n", thisfunc);

  return;
}
