/*
** gptl.cu
** Author: Jim Rosinski
**
** Main file contains most CUDA GPTL functions
*/

#undef CHECK_1SEC
#undef USE_THREADS

#include <stdio.h>
#include <string.h>        /* memcpy */
#include <cuda.h>

#include "./private.h"
#include "./gptl_cuda.h"

#define FLATTEN_TIMERS(SUB1,SUB2) (SUB1)*maxtimers + (SUB2)
#define FLATTEN_HASH(SUB1,SUB2) (SUB1)*tablesize + (SUB2)

__device__ static int *badderef = 0;       // dereferencing this should should crash
__device__ static Hashentry *hashtable;
__device__ static Timer *timers = 0;             /* linked list of timers */
__device__ static void *timersaddr;         // For verifying address of timers stays put
__device__ static Timer **lasttimer = 0;               /* last element in list */
__device__ static int *max_name_len;              /* max length of timer name */
__device__ static int *ntimers_allocated;
__device__ static int maxwarps = -1;              /* max warps */
#ifdef USE_CONSTANT
__device__ __constant__ static int maxtimers;                  /* max timers */
__device__ __constant__ static int tablesize;                  // size of hash table
#else
__device__ static int maxtimers = -1;                  /* max timers */
__device__ static int tablesize;                  // size of hash table
#endif
__device__ static int tablesizem1;                // one less
__device__ static int maxwarpid_found = 0;           /* number of warps found : init to 0 */
__device__ static int maxwarpid_timed = 0;           /* number of warps analyzed : init to 0 */
__device__ static bool disabled = false;          /* Timers disabled? */
__device__ static bool initialized = false;       /* GPTLinitialize has been called */
__device__ static bool verbose = false;           /* output verbosity */
__device__ static double gpu_hz = 0.;             // clock freq
#ifdef CHECK_1SEC
typedef struct {
  long long last;
  long long current;
  long long delta;
  uint smid;
  float diff;
} Startstop;
__device__ static volatile Startstop *startstop;
#endif

extern "C" {

/* Local function prototypes */
#ifdef CHECK_1SEC
__global__ static void initialize_gpu (const int, const int, const int, const int, const double, 
				       Timer *, Hashentry *, int *, int *, Startstop *, Timer **);
#else
__global__ static void initialize_gpu (const int, const int, const int, const int, const double, 
				       Timer *, Hashentry *, int *, int *, Timer **);
#endif
__device__ static inline int get_warp_num (void);         /* get 0-based warp number */
__device__ static inline unsigned int genhashidx (const char *);
__device__ static __forceinline__ Timer *getentry (const int, const char *, const unsigned int);
__device__ static inline int update_stats (Timer *, const long long, const int);
__device__ static int update_ll_hash (Timer *, int, unsigned int);
__device__ static inline int update_ptr (Timer *, const int);
__device__ static __forceinline__ int my_strlen (const char *);
__device__ static inline char *my_strcpy (char *, const char *);
__device__ static __forceinline__ int my_strcmp (const char *, const char *);
__device__ static void init_gpustats (Gpustats *, Timer *, int);
__device__ static void fill_gpustats (Gpustats *, Timer *, int);
__device__ static int gptlstart_sim (const char *, long long);
__device__ static Timer *get_new_timer (int, const char *, const char *);

/* VERBOSE is a debugging ifdef local to the rest of this file */
#define VERBOSE

__host__ int GPTLinitialize_gpu (const int verbose_in,
				 const int maxwarps_in,
				 const int tablesize_in,
				 const int maxtimers_in,
				 const double gpu_hz_in)
{
  size_t nbytes;  // number of bytes to allocate
  static Hashentry *hashtable_cpu;
  static int *max_name_len_cpu;              /* max length of timer name */
  static int *ntimers_allocated_cpu;              /* max length of timer name */
  static Timer *timers_cpu = 0;             /* linked list of timers */
  static Timer **lasttimer_cpu = 0;               /* last element in list */
#ifdef CHECK_1SEC
  static Startstop *startstop_cpu = 0;
#endif
  //  static const char *thisfunc = "GPTLinitialize_gpu";

  nbytes = maxwarps_in * maxtimers_in * sizeof (Timer);
  gpuErrchk (cudaMalloc (&timers_cpu, nbytes));

  nbytes = maxwarps_in * tablesize_in * sizeof (Hashentry);
  gpuErrchk (cudaMalloc (&hashtable_cpu, nbytes));

  nbytes = maxwarps_in * sizeof (int);
  gpuErrchk (cudaMalloc (&max_name_len_cpu, nbytes));
  gpuErrchk (cudaMalloc (&ntimers_allocated_cpu, nbytes));

  nbytes = maxwarps_in * sizeof (Timer *);
  gpuErrchk (cudaMalloc (&lasttimer_cpu, nbytes));

#ifdef CHECK_1SEC
  nbytes = maxwarps_in * sizeof (Startstop);
  gpuErrchk (cudaMalloc (&startstop_cpu, nbytes));
#endif

  // Using constant memory doesn't seem to help much if at all
  // First arg is pass by reference so no "&"
  // maxtimers_in and tablesize_in will be ignored if USE_CONSTANT is defined
#ifdef USE_CONSTANT
  gpuErrchk (cudaMemcpyToSymbol (maxtimers, &maxtimers_in, sizeof (int)));
  gpuErrchk (cudaMemcpyToSymbol (tablesize, &tablesize_in, sizeof (int)));
#endif

  initialize_gpu <<<1,1>>> (verbose_in, tablesize_in, maxwarps_in, maxtimers_in, gpu_hz_in,
			    timers_cpu, 
			    hashtable_cpu, 
			    max_name_len_cpu, 
			    ntimers_allocated_cpu,
#ifdef CHECK_1SEC
			    startstop_cpu,
#endif			    
			    lasttimer_cpu);
  // This should flush any existing print buffers
  cudaDeviceSynchronize ();
  return 0;
}

/*
** GPTLinitialize_gpu (): Initialization routine must be called from single-threaded
**   region before any other timing routines may be called.  The need for this
**   routine could be eliminated if not targetting timing library for threaded
**   capability. 
*/
__global__ static void initialize_gpu (const int verbose_in,
				       const int tablesize_in,
				       const int maxwarps_in,
				       const int maxtimers_in,
				       const double gpu_hz_in,
				       Timer *timers_cpu,
				       Hashentry *hashtable_cpu,
				       int *max_name_len_cpu,
				       int *ntimers_allocated_cpu,
#ifdef CHECK_1SEC
				       Startstop *startstop_cpu,
#endif				       
				       Timer **lasttimer_cpu)
{
  int i, w;           // loop indices over timer, table, warp
  int wi;
  long long t1, t2;      /* returned from underlying timer */
  static const char *thisfunc = "initialize_gpu";

#ifdef VERBOSE
  printf ("Entered %s\n", thisfunc);
#endif
  if (initialized) {
    (void) GPTLerror_1s ("%s: has already been called\n", thisfunc);
    return;
  }

  // Set global vars from input args
  verbose     = verbose_in;
#ifndef USE_CONSTANT
  tablesize   = tablesize_in;
  maxtimers   = maxtimers_in;
#endif
  tablesizem1 = tablesize_in - 1;
  maxwarps    = maxwarps_in;
  gpu_hz = gpu_hz_in;
  timers = timers_cpu;
  timersaddr = (void *) &timers;
  hashtable = hashtable_cpu;
  max_name_len = max_name_len_cpu;
  ntimers_allocated = ntimers_allocated_cpu;
#ifdef CHECK_1SEC
  startstop = startstop_cpu;
#endif
  lasttimer = lasttimer_cpu;

  // Initialize hashtable
  for (w = 0; w < maxwarps; ++w) {
    wi = FLATTEN_HASH(w,0);
    memset (&hashtable[wi], 0, tablesize * sizeof (Hashentry));
    for (i = 0; i < tablesize; ++i) {
      wi = FLATTEN_HASH(w,i);
      hashtable[wi].entry = NULL;
    }
  }

  // Initialize timers, lasttimer
  for (w = 0; w < maxwarps; ++w) {
    max_name_len[w] = 0;
    ntimers_allocated[w] = 0;

    wi = FLATTEN_TIMERS(w,0);
    memset (&timers[wi], 0, maxtimers * sizeof (Timer));
    lasttimer[w] = &timers[wi];
    for (i = 0; i < maxtimers-1; ++i) {
      wi = FLATTEN_TIMERS(w,i);
      timers[wi].next = &timers[wi+1];
    }

    // Make a timer "GPTL_ROOT" to ensure no orphans, and to simplify printing.
    memcpy (lasttimer[w]->name, "GPTL_ROOT", 9+1);
    lasttimer[w]->next = NULL;
  }

  if (verbose) {
    t1 = clock64 ();
    t2 = clock64 ();
    if (t1 > t2)
      printf ("%s: negative delta-t=%lld\n", thisfunc, t2-t1);

    printf ("Per call overhead est. t2-t1=%g should be near zero\n", t2-t1);
    printf ("Underlying wallclock timing routine is clock64\n");
  }

  initialized = true;
  printf("end %s: maxwarps=%d hashtable addr=%p\n", thisfunc, maxwarps, hashtable);
}

/*
** GPTLfinalize_gpu (): Finalization routine must be called from single-threaded
**   region. Free all malloc'd space
*/
__global__ void GPTLfinalize_gpu (void)
{
  static const char *thisfunc = "GPTLfinalize_gpu";

  if ( ! initialized) {
    (void) GPTLerror_1s ("%s: initialization was not completed\n", thisfunc);
    return;
  }

  cudaFree (hashtable);
  cudaFree (timers);
  cudaFree (lasttimer);
  cudaFree (max_name_len);
  
  GPTLreset_errors_gpu ();

  /* Reset initial values */
  timers = 0;
  lasttimer = 0;
  max_name_len = 0;
  disabled = false;
  initialized = false;
  verbose = false;
}

/*
** GPTLstart_gpu: start a timer
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
  int wi;
  unsigned int indx; /* hash table index */
  static const char *thisfunc = "GPTLstart_gpu";

  //JR: This is for debugging the CUDA bug in which static data reverts to initial values
  //  printf("%s: name=%s: maxwarps=%d hashtable addr=%p\n", thisfunc, name, maxwarps, hashtable);

  if (disabled)
    return SUCCESS;

  if ( ! initialized)
    return GPTLerror_2s ("%s name=%s: GPTLinitialize_gpu has not been called\n", thisfunc, name);

  w = get_warp_num ();

  // Return if not thread 0 of the warp, or warpId is outside range of available timers
  if (w == NOT_ROOT_OF_WARP || w == WARPID_GT_MAXWARPS)
    return SUCCESS;

  /* ptr will point to the requested timer in the current list, or NULL if this is a new entry */
  indx = genhashidx (name);
  ptr = getentry (w, name, indx);

  /* 
  ** Recursion => increment depth in recursion and return.  We need to return 
  ** because we don't want to restart the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */
  if (ptr && ptr->onflg) {
    ++ptr->recurselvl;
    return SUCCESS;
  }

  if ( ! ptr) { // Add a new entry and initialize, first ensuring that a collision has not happened
    wi = FLATTEN_HASH(w,indx);
    if (hashtable[wi].entry) 
      return GPTLerror_3s ("%s: Collision: existing timer=%s new timer=%s: Ignoring new timer\n", 
			   thisfunc, hashtable[wi].entry->name, name);

    if ((ptr = get_new_timer (w, name, thisfunc)) == NULL)
      return GPTLerror_2s ("%s: get_new_timer failure for timer %s\n", thisfunc, name);

    if (update_ll_hash (ptr, w, indx) != 0)
      return GPTLerror_1s ("%s: update_ll_hash error\n", thisfunc);
  }

  if (update_ptr (ptr, w) != 0)
    return GPTLerror_1s ("%s: update_ptr error\n", thisfunc);

  return SUCCESS;
}

/*
** GPTLinit_handle_gpu: Initialize a handle for further use by GPTLstart_handle() and GPTLstop_handle()
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
    return SUCCESS;

  *handle = (int) genhashidx (name);
  return SUCCESS;
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
  int wi;
  static const char *thisfunc = "GPTLstart_handle_gpu";

  if (disabled)
    return SUCCESS;

  if ( ! initialized)
    return GPTLerror_2s ("%s name=%s: GPTLinitialize_gpu has not been called\n", 
			 thisfunc, name);

  w = get_warp_num ();

  // Return if not thread 0 of the warp, or warpId is outside range of available timers
  if (w == NOT_ROOT_OF_WARP || w == WARPID_GT_MAXWARPS)
    return SUCCESS;

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

  ptr = getentry (w, name, (unsigned int) *handle);
  
  /* 
  ** Recursion => increment depth in recursion and return.  We need to return 
  ** because we don't want to restart the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */
  if (ptr && ptr->onflg) {
    ++ptr->recurselvl;
    return SUCCESS;
  }

  if ( ! ptr) { // Add a new entry and initialize, first ensuring that a collision has not happened
    wi = FLATTEN_HASH(w,*handle);
    if (hashtable[wi].entry) 
      return GPTLerror_3s ("%s: Collision: existing timer=%s new timer=%s: Ignoring new timer\n", 
			   thisfunc, hashtable[wi].entry->name, name);

    if ((ptr = get_new_timer (w, name, thisfunc)) == NULL)
      return GPTLerror_2s ("%s: get_new_timer failure for timer %s\n", thisfunc, name);

    if (update_ll_hash (ptr, w, (unsigned int) *handle) != 0)
      return GPTLerror_1s ("%s: update_ll_hash error\n", thisfunc);
  }

  if (update_ptr (ptr, w) != 0)
    return GPTLerror_1s ("%s: update_ptr error\n", thisfunc);

  return SUCCESS;
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
  int wi;
  static const char *thisfunc = "update_ll_hash";

  nchars = my_strlen (ptr->name);
  if (nchars > max_name_len[w])
    max_name_len[w] = nchars;

  wi = FLATTEN_HASH(w,indx);
  hashtable[wi].entry = ptr;

#ifdef CHECK_SM
  uint smid;
  asm ("mov.u32 %0, %smid;" : "=r"(smid));
  //  printf ("%s: name=%s warp= %d sm= %d\n", thisfunc, ptr->name, w, smid);
  ptr->smid = smid;
#endif

#ifdef DEBUG
  printf("%s: name=%s indx=%d wi=%d\n", thisfunc, ptr->name, indx, wi);
#endif
  return SUCCESS;
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
  long long delta;
  static const char *thisfunc = "update_ptr";

#ifdef DEBUG
  printf ("%s: ptr=%p setting onflg=true\n", thisfunc, ptr);
#endif

#ifdef CHECK_SM
  uint smid;
  asm ("mov.u32 %0, %smid;" : "=r"(smid));
  if (smid != ptr->smid) {
    //    printf ("%s: name=%s warp=%d sm changed from %d to %d\n", 
    //	    thisfunc, ptr->name, w, ptr->smid, smid);
    ptr->smid = smid;
  }
#endif
  
  ptr->onflg = true;
  tp2 = clock64 ();
  delta = (tp2 - ptr->wall.last);
  if (delta < 0) {
    ++ptr->negcount;
    printf ("GPTL: %s name=%s w=%d WARNING: backward by %g sec: resetting anyway \n",
	    thisfunc, ptr->name, w, delta/-gpu_hz);
    if ((void *) &timers != timersaddr) {
      printf ("%s: timers changed address!!!! old=%p new=%p\n", thisfunc, &timers, timersaddr);
      return *badderef;
    }
    if (my_strcmp (timers[0].name, "GPTL_ROOT") != 0) {
      printf ("%s: timers[0].name=%s should=GPTL_ROOT\n", thisfunc, timers[0].name);
      return *badderef;
    }
  }
  ptr->wall.last = tp2;
  return SUCCESS;
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
  volatile long long tp1;             /* time stamp */
  Timer *ptr;                /* linked list pointer */
  int w;                     /* warp number for this process */
  unsigned int indx;         /* index into hash table */
  static const char *thisfunc = "GPTLstop_gpu";

  if (disabled)
    return SUCCESS;

  if ( ! initialized)
    return GPTLerror_1s ("%s: GPTLinitialize_gpu has not been called\n", thisfunc);

  /* Get the timestamp */
  tp1 = clock64 ();

  w = get_warp_num ();

  // Return if not thread 0 of the warp, or warpId is outside range of available timers
  if (w == NOT_ROOT_OF_WARP || w == WARPID_GT_MAXWARPS)
    return SUCCESS;

  indx = genhashidx (name);
  if (! (ptr = getentry (w, name, indx)))
    return GPTLerror_1s1d1s ("%s warp %d: timer for %s had not been started.\n",
			     thisfunc, w, name);

  if ( ! ptr->onflg )
    return GPTLerror_2s ("%s: timer %s was already off.\n", 
			 thisfunc, ptr->name);
  /* 
  ** Recursion => decrement depth in recursion and return.  We need to return
  ** because we don't want to stop the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */
  if (ptr->recurselvl > 0) {
    --ptr->recurselvl;
    ++ptr->count;
    return SUCCESS;
  }

  if (update_stats (ptr, tp1, w) != 0)
    return GPTLerror_1s ("%s: error from update_stats\n", thisfunc);

  return SUCCESS;
}

/*
** GPTLstop_handle_gpu: stop a timer based on a handle
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
  volatile long long tp1;         /* time stamp */
  Timer *ptr;                /* linked list pointer */
  int w;                     /* warp number for this process */
  unsigned int indx;
  static const char *thisfunc = "GPTLstop_handle_gpu";

  if (disabled)
    return SUCCESS;

  if ( ! initialized)
    return GPTLerror_1s ("%s: GPTLinitialize_gpu has not been called\n", thisfunc);

  /* Get the timestamp */
  tp1 = clock64 ();

  w = get_warp_num ();

  // Return if not thread 0 of the warp, or warpId is outside range of available timers
  if (w == NOT_ROOT_OF_WARP || w == WARPID_GT_MAXWARPS)
    return SUCCESS;

  indx = (unsigned int) *handle;
  if (indx == 0 || indx > tablesizem1)
    return GPTLerror_1s1d1s ("%s: bad input handle=%u for timer %s.\n", 
			     thisfunc, (int) indx, name);
  
  if ( ! (ptr = getentry (w, name, indx)))
    return GPTLerror_1s1d1s ("%s: handle=%u has not been set for timer %s.\n", 
			     thisfunc, (int) indx, name);

  if ( ! ptr->onflg )
    return GPTLerror_2s ("%s: timer %s was already off.\n", thisfunc, ptr->name);

  /* 
  ** Recursion => decrement depth in recursion and return.  We need to return
  ** because we don't want to stop the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */
  if (ptr->recurselvl > 0) {
    --ptr->recurselvl;
    ++ptr->count;
    return SUCCESS;
  }

  if (update_stats (ptr, tp1, w) != 0)
    return GPTLerror_1s ("%s: error from update_stats\n", thisfunc);

  return SUCCESS;
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
  uint smid;         // only needed when certain ifdefs defined
  long long delta;   /* difference */
  static const char *thisfunc = "update_stats";
#ifdef DEBUG
  printf ("%s: ptr=%p setting onflg=false\n", thisfunc, ptr);
#endif

  ptr->onflg = false;
  delta = tp1 - ptr->wall.last;

  if (delta < 0) {
    ++ptr->negcount;
    printf ("GPTL: %s name=%s w=%d WARNING NEGATIVE DELTA ENCOUNTERED: %lld-%lld=%lld: IGNORING\n", 
	    thisfunc, ptr->name, w, tp1, ptr->wall.last, delta);

    // If either of these tests fail, need to abort
    if ((void *) &timers != timersaddr) {
      printf ("%s: timers changed address!!!! old=%p new=%p\n", thisfunc, &timers, timersaddr);
      return *badderef;
    }
    if (my_strcmp (timers[0].name, "GPTL_ROOT") != 0) {
      printf ("%s: timers[0].name=%s should=GPTL_ROOT\n", thisfunc, timers[0].name);
      return *badderef;
    }
    return SUCCESS;  // Return without adding the bad delta
  }

  ++ptr->count;
  ptr->wall.accum += delta;

  if (ptr->count == 1) {
    ptr->wall.max = delta;
    ptr->wall.min = delta;
  } else {
    if (delta > ptr->wall.max)
      ptr->wall.max = delta;
    if (delta < ptr->wall.min)
      ptr->wall.min = delta;
  }

#ifdef CHECK_1SEC
  double diff =  delta / gpu_hz;
  if (diff < 0.99) {
    asm ("mov.u32 %0, %smid;" : "=r"(smid));
    printf ("%s: w=%d smid=%u smid_sleep=%u\n", thisfunc, w, smid, startstop[w].smid);
    printf ("start(gptl) %lld start(sleep) %lld stop(sleep) %lld stop(gptl) %lld\n",
	    ptr->wall.last, startstop[w].last, startstop[w].current, tp1);
    printf ("diff(gptl) %f diff(sleep) %f\n", delta/gpu_hz, startstop[w].diff);
  }
#endif

#ifdef CHECK_SM
  asm ("mov.u32 %0, %smid;" : "=r"(smid));
  if (smid != ptr->smid) {
    printf ("%s: name=%s warp=%d sm changed from %d to %d\n", 
	    thisfunc, ptr->name, w, ptr->smid, smid);
    ptr->smid = smid;
  }
#endif
  return SUCCESS;
}

/*
** GPTLenable_gpu: enable timers
*/
__global__ void GPTLenable_gpu (void)
{
  disabled = false;
}

/*
** GPTLdisable_gpu: disable timers
*/
__global__ void GPTLdisable_gpu (void)
{
  disabled = true;
}

/*
** GPTLreset_gpu: reset all timers to 0
**
** Return value: 0 (success) or GPTLerror (failure)
*/
__global__ void GPTLreset_gpu (void)
{
  int w;             /* index over warps */
  int wi;
  Timer *ptr;        /* linked list index */
  static const char *thisfunc = "GPTLreset_gpu";

  if ( ! initialized) {
    (void) GPTLerror_1s ("%s: GPTLinitialize_gpu has not been called\n", thisfunc);
    return;
  }

  for (w = 0; w <= maxwarpid_timed; w++) {
    wi = FLATTEN_TIMERS(w,0);
    for (ptr = &timers[wi]; ptr; ptr = ptr->next) {
      ptr->onflg = false;
      ptr->count = 0;
      memset (&ptr->wall, 0, sizeof (ptr->wall));
    }
  }

  if (verbose)
    printf ("%s: accumulators for all GPU timers set to zero\n", thisfunc);
}

/*
** genhashidx: generate hash index
**
** Input args:
**   name: string to be hashed on
**
** Return value: hash value
*/
__device__ static inline unsigned int genhashidx (const char *name)
{
  const unsigned char *c;       /* pointer to elements of "name" */
  unsigned int indx;            /* return value of function */
  unsigned int mididx, lastidx; /* mid and final index of name */

  lastidx = my_strlen (name) - 1;
  mididx = lastidx / 2;
  /* 
  ** Disallow a hash index of zero (by adding 1 at the end) since user input of an uninitialized 
  ** value, though an error, has a likelihood to be zero.
  */
  c = (unsigned char *) name;
  indx = (MAX_CHARS*c[0] + (MAX_CHARS-mididx)*c[mididx] + 
	  (MAX_CHARS-lastidx)*c[lastidx]) % tablesizem1 + 1;
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
__device__ static __forceinline__ Timer *getentry (const int w,             // warp number
						   const char *name,        // name to hash
						   const unsigned int indx) // hash index
{
  Timer *ret = NULL;
  const int wi = FLATTEN_HASH(w,indx);

  if (hashtable[wi].entry && STRMATCH (name, hashtable[wi].entry->name))
    ret = hashtable[wi].entry;
  return ret;
}

/*
** placebo: does nothing and returns zero always. Useful for estimating overhead costs
*/
__device__ static int init_placebo ()
{
  return SUCCESS;
}

__device__ static inline long long utr_placebo ()
{
  return (long long) SUCCESS;
}

__device__ static inline int get_warp_num ()
{
  int warpId;
  int blockId = blockIdx.x 
    + blockIdx.y * gridDim.x 
    + gridDim.x * gridDim.y * blockIdx.z; 
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
    + (threadIdx.z * (blockDim.x * blockDim.y))
    + (threadIdx.y * blockDim.x)
    + threadIdx.x;

#if 0
  printf ("gridDim= %d %d, blockDim= %d %d %d blockIdx= %d %d %d, threadIdx= %d %d %d "
	  "myblockId= %d mythreadId= %d\n",	  
	  gridDim.x, gridDim.y, 
	  blockDim.x, blockDim.y, blockDim.z, 
	  blockIdx.x, blockIdx.y, blockIdx.z, 
	  threadIdx.x, threadIdx.y, threadIdx.z,
	  blockId, threadId);
#endif
  
  // Only thread 0 of the warp will be timed
  if (threadId % WARPSIZE != 0)
    return NOT_ROOT_OF_WARP;

  // USE_THREADS means use threadId not warpId 
#ifdef USE_THREADS
  warpId = threadId;
#else
  warpId = threadId / WARPSIZE;
#endif

  // maxwarpid_found is needed only by CPU code when printing results
  if (warpId+1 > maxwarpid_found)
    maxwarpid_found = warpId;

  if (warpId > maxwarps-1)
    return WARPID_GT_MAXWARPS;

  // if we get here we have a usable warpId
  if (warpId > maxwarpid_timed)
    maxwarpid_timed = warpId;

  return warpId;
}

__global__ void GPTLget_gpusizes (int *maxwarpid_found_out, int *maxwarpid_timed_out)
{
  *maxwarpid_found_out = maxwarpid_found;
  *maxwarpid_timed_out = maxwarpid_timed;
}

__device__ int GPTLget_wallclock_gpu (const char *timername,
				      double *accum, double *max, double *min)
{
  Timer *ptr;          // linked list pointer
  int w;               // warp index
  unsigned int indx;   // hash index returned from getentry (unused)
  static const char *thisfunc = "GPTLget_wallclock_gpu";
  
  if ( ! initialized)
    (void) GPTLerror_1s ("%s: GPTLinitialize_gpu has not been called\n", thisfunc);

  if (gpu_hz == 0.)
    (void) GPTLerror_1s ("%s: gpu_hz has not been set\n", thisfunc);

  w = get_warp_num ();
  if (w == NOT_ROOT_OF_WARP || w == WARPID_GT_MAXWARPS)
    return SUCCESS;
  
  indx = genhashidx (timername);
  ptr = getentry (w, timername, indx);
  if ( ! ptr)
    return GPTLerror_2s ("%s: requested timer %s does not exist\n", thisfunc, timername);

  *accum = ptr->wall.accum / gpu_hz;
  *max   = ptr->wall.max   / gpu_hz;
  *min   = ptr->wall.min   / gpu_hz;
  return 0;
}

__device__ static Timer *get_new_timer (int w, const char *name, const char *caller)
{
  int numchars;
  Timer *ptr = NULL;
#ifdef DEBUG
  static const char *thisfunc = "get_new_timer";
#endif
  
  if (w > maxwarps-1) {
    (void) GPTLerror_1s2d ("%s: w=%d exceeds maxwarps=%d\n", caller, w, maxwarps);
    return ptr;
  }

  if (ntimers_allocated[w] > maxtimers-2) {
    (void) GPTLerror_2s2d ("%s: name=%s warp=%d: ntimers allocated=%d exceeds maxtimers-2: CANT start new timer\n", 
			   caller, name, w, ntimers_allocated[w]);
    return ptr;
  }

  ptr = lasttimer[w]++;
  ptr->next = lasttimer[w];
  ptr = ptr->next;
  ptr->next = NULL;
  numchars = MIN (my_strlen (name), MAX_CHARS);
  memcpy (ptr->name, name, numchars);
  ptr->name[numchars] = '\0';
#ifdef DEBUG
  printf ("%s: name=%s w=%d added at position %d\n", thisfunc, ptr->name, w, ntimers_allocated[w]);
#endif
  ++ntimers_allocated[w];
  return ptr;
}
//JR want to use variables to dimension arrays but nvcc is not C99 compliant
__global__ void GPTLfill_gpustats (Gpustats *gpustats, 
				   int *max_name_len_out,
				   int *ngputimers)
{
  int w,ww;          // warp indices
  int wi, wwi;
  int n;             // timer index
  Timer *ptr, *tptr; // loop through linked list
  static const char *thisfunc = "GPTLfill_gpustats";

  // Step 0: initialize "beenprocessed" flag to false everywhere
  // Also: determine max_name_len
  *max_name_len_out = 0;
  for (w = 0; w <= maxwarpid_timed; ++w) {
    if (max_name_len[w] > *max_name_len_out)
      *max_name_len_out = max_name_len[w];
    wi = FLATTEN_TIMERS(w,0);
    for (ptr = timers[wi].next; ptr; ptr = ptr->next) {
      ptr->beenprocessed = false;
    }
  }

  // Step 1: process entries for all warps based on those in warp 0
  n = 0;
  for (ptr = timers[0].next; ptr; ptr = ptr->next, ++n) {
    if (n > maxtimers-1) {
      printf ("%s: Timer=%s: Truncating timer results at n=%d name=%s:" 
	      "Increase maxtimers\n", thisfunc, n, ptr->name);
      *ngputimers = n;
      return;
    }
    w = 0;
    init_gpustats (&gpustats[n], ptr, w);
    for (w = 1; w <= maxwarpid_timed; ++w) {
      wi = FLATTEN_TIMERS(w,0);
      for (tptr = timers[wi].next; tptr && my_strcmp (ptr->name, tptr->name); 
	   tptr = tptr->next);
      if (tptr) {     // my_strcmp must have hit a match
	fill_gpustats (&gpustats[n], tptr, w);
	tptr->beenprocessed = true;
      }
    }
  }

  // Step 2: process entries which do not exist in warp 0
  for (w = 1; w <= maxwarpid_timed; ++w) {
    wi = FLATTEN_TIMERS(w,0);
    for (ptr = timers[wi].next; ptr; ptr = ptr->next) {
      if ( ! ptr->beenprocessed) {
	++n;           // found a new timer name which has not yet been processed (not in warp 0)
	if (n > maxtimers-1) {
	  printf ("%s: Timer=%s: Truncating timer results at n=%d name=%s: "
		  "Increase maxtimers\n", thisfunc, n, ptr->name);
	  *ngputimers = n;
	  return;
	}
	init_gpustats (&gpustats[n], ptr, w);
	printf ("%s: Found non-root entry for name=%s at warp=%d\n", 
		thisfunc, ptr->name, w);
	for (ww = w+1; ww <= maxwarpid_timed; ++w) {
	  wwi = FLATTEN_TIMERS(ww,0);
	  for (tptr = timers[wwi].next; tptr && my_strcmp (ptr->name, tptr->name); 
	       tptr = tptr->next);
	  if (tptr) {  // my_strcmp must have hit a match
	    if ( tptr->beenprocessed) {
	      printf ("%s: Something fishy: %s from warp=%d not processed "
		      "but %s from warp=%d has been\n",
		      thisfunc, ptr->name, w, tptr->name, ww);
	    } else {
	      printf ("%s: Found additional entry for name=%s at warp=%d\n", 
		      thisfunc, tptr->name, ww);
	      fill_gpustats (&gpustats[n], tptr, ww);
	      tptr->beenprocessed = true;
	    }
	  }
	}
      }
    }
  }

  // Step 3: Verify all timers have been processed (if MAX_GPUTIMERS limit not exceeded)
  for (w = 0; w <= maxwarpid_timed; ++w) {
    wi = FLATTEN_TIMERS(w,0);
    for (ptr = timers[wi].next; ptr; ptr = ptr->next) {
      if ( ! ptr->beenprocessed) {
	printf ("%s: Something fishy: Timer=%s was not processed\n", 
		thisfunc, ptr->name);
      }
    }
  }

  *ngputimers = n;

#ifdef DEBUG
  printf ("%s: ngputimers=%d\n", thisfunc, n);
  for (n = 0; n < *ngputimers; ++n) {
    printf ("%s: timer=%s accum_max=%lld accum_min=%lld count_max=%d nwarps=%d\n", 
	    thisfunc, gpustats[n].name, gpustats[n].accum_max, gpustats[n].accum_min, gpustats[n].count_max, gpustats[n].nwarps);
  }
#endif
  return;
}

__device__ static void init_gpustats (Gpustats *gpustats, Timer *ptr, int w)
{
  (void) my_strcpy (gpustats->name, ptr->name);
  gpustats->count  = ptr->count;
  gpustats->nwarps = 1;

  gpustats->accum_max = ptr->wall.accum;
  gpustats->accum_max_warp = w;

  gpustats->accum_min = ptr->wall.accum;
  gpustats->accum_min_warp = w;

  gpustats->count_max = ptr->count;
  gpustats->count_max_warp = w;

  gpustats->count_min = ptr->count;
  gpustats->count_min_warp = w;

  gpustats->negcount_max = ptr->negcount;
  gpustats->negcount_max_warp = w;

  ptr->beenprocessed = true;
}

__device__ static void fill_gpustats (Gpustats *gpustats, Timer *ptr, int w)
{
  gpustats->count += ptr->count;
  ++gpustats->nwarps;

  if (ptr->wall.accum > gpustats->accum_max) {
    gpustats->accum_max = ptr->wall.accum;
    gpustats->accum_max_warp = w;
  }

  if (ptr->wall.accum < gpustats->accum_min) {
    gpustats->accum_min = ptr->wall.accum;
    gpustats->accum_min_warp = w;
  }
  
  if (ptr->count > gpustats->count_max) {
    gpustats->count_max = ptr->count;
    gpustats->count_max_warp = w;
  }
	
  if (ptr->count < gpustats->count_min) {
    gpustats->count_min = ptr->count;
    gpustats->count_min_warp = w;
  }

  if (ptr->negcount > gpustats->negcount_max) {
    gpustats->negcount_max = ptr->negcount;
    gpustats->negcount_max_warp = w;
  }
}

__device__ static __forceinline__ int my_strlen (const char *str)
{
#ifdef SLOW
  int i;
  for (i = 0; str[i] != '\0'; ++i);
  return i;
#else
  const char *s;
  for (s = str; *s; ++s);
  return(s - str);
#endif
}

__device__ static inline char *my_strcpy (char *dest, const char *src)
{
  char *ret = dest;

  while (*src != '\0')
    *dest++ = *src++;
  *dest = '\0';
  return ret;
}

//JR Both of these have about the same performance
__device__ static __forceinline__ int my_strcmp (const char *str1, const char *str2)
{
#ifdef MINE
  while (*str1 == *str2) {
    if (*str1 == '\0')
      break;
    ++str1;
    ++str2;
  }
  return (int) (*str1 - *str2);
#else
  register const unsigned char *s1 = (const unsigned char *) str1;
  register const unsigned char *s2 = (const unsigned char *) str2;
  register unsigned char c1, c2;
 
  do {
      c1 = (unsigned char) *s1++;
      c2 = (unsigned char) *s2++;
      if (c1 == '\0')
	return c1 - c2;
  } while (c1 == c2); 
  return c1 - c2;
#endif
}

// Overhead estimate functions start here
/*
** GPTLget_overhead: return current status info about a timer. If certain stats are not enabled, 
** they should just have zeros in them.
** 
** Output args:
**   get_warp_num_ohd: Getting my warp index
**   genhashidx_ohd:     Generating hash index
**   getentry_ohd:       Finding entry in hash table
**   utr_ohd:            Underlying timer routine
**   self_ohd:           Estimate of GPTL-induced overhead in the timer itself (included in "Wallclock")
**   parent_ohd:         Estimate of GPTL-induced overhead for the timer which appears in its parents
*/
__global__ void GPTLget_overhead_gpu (long long *ftn_ohd,
				      long long *get_warp_num_ohd, // Getting my warp index
				      long long *genhashidx_ohd,   // Generating hash index
				      long long *getentry_ohd,     // Finding entry in hash table
				      char *getentry_ohd_name,     // name used for getentry
				      long long *utr_ohd,          // Underlying timing routine
				      long long *self_ohd,
				      long long *parent_ohd,
				      long long *my_strlen_ohd,
				      long long *STRMATCH_ohd)
{
  long long t1, t2;          /* Initial, final timer values */
  int i;
  int ret;
  int mywarp;                /* which warp are we */
  unsigned int hashidx;      /* Hash index */
  long long nchars;
  Timer *entry;              /* placeholder for return from "getentry()" */
  char name[MAX_CHARS+1];
  static char *timername = "timername";

  /*
  ** Gather timings by running kernels 1000 times each
  ** First: Fortran wrapper overhead
  */
  nchars = my_strlen (timername);
  t1 = clock64();
  for (i = 0; i < 1000; ++i) {
    /* 9 is the number of characters in "timername" */
    ret = gptlstart_sim (timername, nchars);
  }
  t2 = clock64();
  *ftn_ohd = (t2 - t1) / 1000;

  /* get_warp_num() overhead */
  t1 = clock64();
  for (i = 0; i < 1000; ++i) {
    mywarp = get_warp_num ();
  }
  t2 = clock64();
  get_warp_num_ohd[0] = (t2 - t1) / 1000;

  /* genhashidx overhead */
  t1 = clock64();
  for (i = 0; i < 1000; ++i) {
    hashidx = genhashidx (timername);
  }
  t2 = clock64();
  *genhashidx_ohd = (t2 - t1) / 1000;

  /* 
  ** getentry overhead
  ** Even if there are no user timers, GPTL_ROOT can be used
  ** 
  */
  if (timers[0].next)
    my_strcpy (name, timers[0].next->name); // first user entry
  else
    my_strcpy (name, timers[0].name);       // GPTL_ROOT

  hashidx = genhashidx (name);
  t1 = clock64();
  for (i = 0; i < 1000; ++i) {
    entry = getentry (0, name, hashidx);
  }
  t2 = clock64();

  *getentry_ohd = (t2 - t1) / 1000;
  my_strcpy (getentry_ohd_name, name);

  /* utr overhead */
  t1 = clock64();
  for (i = 0; i < 1000; ++i) {
    t2 = clock64();
  }
  *utr_ohd = (t2 - t1) / 1000;

  *self_ohd   = *utr_ohd;
  *parent_ohd = *utr_ohd + 2*(*ftn_ohd + *get_warp_num_ohd + 
			      *genhashidx_ohd + *getentry_ohd);

  // my_strlen overhead
  t1 = clock64();
  for (i = 0; i < 1000; ++i) {
    ret = my_strlen (timername);
  }
  t2 = clock64();
  *my_strlen_ohd = (t2 - t1) / 1000;

  // STRMATCH overhead
  t1 = clock64();
  for (i = 0; i < 1000; ++i) {
    // Use wi computed above
    ret = STRMATCH (name, "ZZZ");
  }
  t2 = clock64();
  *STRMATCH_ohd = (t2 - t1) / 1000;
  return;
}

/*
** gptlstart_sim: Simulate the cost of Fortran wrapper layer "gptlstart()"
** 
** Input args:
**   name: timer name
**   nc:  number of characters in "name"
*/
__device__ static int gptlstart_sim (const char *name, long long nc)
{
  char cname[MAX_CHARS+1];

  if (nc > MAX_CHARS)
    printf ("%d exceeds MAX_CHARS=%d\n", nc, MAX_CHARS);

  for (int n = 0; n < nc; ++n)
    cname[n] = name[n];
  cname[nc] = '\0';
  return SUCCESS;
}

__global__ void GPTLget_memstats_gpu (float *hashmem, float *regionmem)
{
  *hashmem =   (float) maxwarps * tablesize * sizeof (Hashentry);  // fixed size of table
  *regionmem = (float) maxwarps * maxtimers * sizeof (Timer);
  return;
}

__device__ int GPTLmy_sleep (float seconds)
{
#ifdef CHECK_1SEC
  int warpId;
  int blockId = blockIdx.x 
    + blockIdx.y * gridDim.x 
    + gridDim.x * gridDim.y * blockIdx.z; 
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
    + (threadIdx.z * (blockDim.x * blockDim.y))
    + (threadIdx.y * blockDim.x)
    + threadIdx.x;
  int w = threadId / WARPSIZE;
#endif

  volatile long long start, now;
  volatile double delta;
  static const char *thisfunc = "GPTLmy_sleep";

  if (gpu_hz == 0.)
    return GPTLerror_1s ("%s: need to set gpu_hz via call to GPTLinitialize_gpu() first\n",
			 thisfunc);

  start = clock64();
  do {
    now = clock64();
    delta = (now - start) / gpu_hz;
  } while (delta < seconds);

#ifdef CHECK_1SEC
  if (threadId % WARPSIZE == 0) {
    uint smid;
    startstop[w].last = start;
    startstop[w].current = now;
    startstop[w].delta = now - start;
    asm ("mov.u32 %0, %smid;" : "=r"(smid));
    startstop[w].smid = smid;
    startstop[w].diff = (now - start) / gpu_hz;
    if (startstop[w].diff < 0.99 || startstop[w].diff > 1.01)
      printf ("%s: diff %f start %lld now %lld\n", thisfunc, startstop[w].diff, start, now);
  }
#endif
  // For some reason, w/o syncthreads, ACC tests often sleep much less than 1 sec
  // But CUDA tests all seem to work fine
  // __syncthreads();
  return SUCCESS;
}

__device__ void GPTLdummy_gpu (int num)
{
  Hashentry x;
  static const char *thisfunc = "GPTLdummy_gpu";
  printf ("%s: num=%d hashtable=%p\n", thisfunc, num, hashtable);
  // If hashtable pointer has become bad, force a cuda error
  x = *hashtable;
  return;
}

}
