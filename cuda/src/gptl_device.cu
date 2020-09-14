/*
** gptl.cu
** Author: Jim Rosinski
**
** Main file contains most CUDA GPTL functions
*/

#include "config.h" // Must be first include.

#include <stdio.h>
#include <string.h>        // memcpy
#include <stdint.h>        // uint types
#include <cuda.h>

#include "device.h"
#include "gptl_cuda.h"

#define FLATTEN_TIMERS(SUB1,SUB2) (SUB1)*maxtimers + (SUB2)

__device__ static Timer *timers = 0;            // array (also linked list) of timers
__device__ static Timername *timernames;        // array of timer names
__device__ static int max_name_len;             // max length of timer name
__device__ static int ntimers = 0;              // number of timers
__device__ __constant__ static int maxtimers;   // max number of timers
__device__ static int maxwarps = -1;            // max warps
__device__ static int maxwarpid_found = 0;      // number of warps found : init to 0
__device__ static bool initialized = false;     // GPTLinitialize has been called
__device__ static bool verbose = false;         // output verbosity
__device__ static double gpu_hz = 0.;           // clock freq
__device__ int warpsize = 0;                    // warp size
__device__ static volatile int mutex = 0;       // critical section unscrambles printf output

extern "C" {

// Local function prototypes
__global__ static void initialize_gpu (const int, const int, const double, Timer *,
				       Timername *, const int);
__device__ static inline int get_warp_num (void);         // get 0-based 1d warp number
__device__ static inline int update_stats_gpu (const int, Timer *, const long long, const int,
					       const uint);
__device__ static int my_strlen (const char *);
__device__ static char *my_strcpy (char *, const char *);
__device__ static int my_strcmp (const char *, const char *);
__device__ static void start_misc (int, const int);
__device__ static void stop_misc (int w, const int handle);
__device__ static void init_gpustats (Gpustats *, int);
__device__ static void fill_gpustats (Gpustats *, int, int);
// Defining PRINTNEG will print to stdout whenever a negative interval (stop minus start) is
// encountered. Only useful when non-zero negative intervals are reported in timing output
// Should be turned OFF normally--very expensive even when no negatives found.
#undef PRINTNEG
#ifdef PRINTNEG
__device__ static void prbits8 (uint64_t);
#endif

/* VERBOSE is a debugging ifdef local to the rest of this file */
#undef VERBOSE

__host__ int GPTLinitialize_gpu (const int verbose_in,
				 const int maxwarps_in,
				 const int maxtimers_in,
				 const double gpu_hz_in,
				 const int warpsize_in)
{
  size_t nbytes;  // number of bytes to allocate

  // Issue cudaMalloc from CPU, and pass address to GPU to avoid mem problems: When run from
  // __global__ routine, mallocable memory is severely decreased for some reason.
  static Timer *timers_cpu = 0;          // array of timers
  static Timername *timernames_cpu = 0; // array of timer names

  // Set constant memory values: First arg is pass by reference so no "&"
  gpuErrchk (cudaMemcpyToSymbol (maxtimers,   &maxtimers_in,    sizeof (int)));

  nbytes = maxwarps_in * maxtimers_in * sizeof (Timer);
  gpuErrchk (cudaMalloc (&timers_cpu, nbytes));

  nbytes =               maxtimers_in * sizeof (Timername);
  gpuErrchk (cudaMalloc (&timernames_cpu, nbytes));

  initialize_gpu <<<1,1>>> (verbose_in,
			    maxwarps_in,
			    gpu_hz_in,
			    timers_cpu,
			    timernames_cpu,
			    warpsize_in);
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
				       const int maxwarps_in,
				       const double gpu_hz_in,
				       Timer *timers_cpu,
				       Timername *timernames_cpu,
				       const int warpsize_in)
{
  int w, wi;        // warp, flattened indices
  long long t1, t2; // returned from underlying timer
  static const char *thisfunc = "initialize_gpu";

#ifdef VERBOSE
  printf ("Entered %s\n", thisfunc);
#endif
  if (initialized) {
    (void) GPTLerror_1s ("%s: has already been called\n", thisfunc);
    return;
  }

  // Set global vars from input args
  verbose    = verbose_in;
  maxwarps   = maxwarps_in;
  gpu_hz     = gpu_hz_in;
  warpsize   = warpsize_in;
  timers     = timers_cpu;
  timernames = timernames_cpu;

  // Initialize timers
  ntimers = 0;
  max_name_len = 0;
  for (w = 0; w < maxwarps; ++w) {
    wi = FLATTEN_TIMERS(w,0);
    memset (&timers[wi], 0, maxtimers * sizeof (Timer));
  }
  // Make a timer "GPTL_ROOT" to ensure no orphans, and to simplify printing.
  memcpy (timernames[0].name, "GPTL_ROOT", 9+1);

  if (verbose) {
    t1 = clock64 ();
    t2 = clock64 ();
    if (t1 > t2)
      printf ("GPTL %s: negative delta-t=%lld\n", thisfunc, t2-t1);

    printf ("Per call overhead est. t2-t1=%g should be near zero\n", (float) (t2-t1));
    printf ("Underlying wallclock timing routine is clock64\n");
  }

  initialized = true;
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

  free (timers);
  free (timernames);
  
  GPTLreset_errors_gpu ();

  // Reset initial values
  timers = 0;
  timernames = 0;
  max_name_len = 0;
  initialized = false;
  verbose = false;
}

/*
** GPTLinit_handle_gpu: Initialize a handle for further use by GPTLstart_gpu() and GPTLstop_gpu()
**
** Input arguments:
**   name: timer name
**
** Output arguments:
**   handle: Index into array for "name"
**
** Return value: 0 (success) or GPTLerror (failure)
*/
__device__ int GPTLinit_handle_gpu (const char *name,     // timer name
				    int *handle)          // handle
{
  int numchars;      // length of "name"
  int mywarp;        // my warp number
  int i;
  static const char *thisfunc = "GPTLinit_handle_gpu";

  // Guts of this function are run only by thread 0 of warp 0 due to loop over warps below. 
  // Need to have each timer have the same index for all warps.
  // Nice feature: Can be called by just thread 0 of warp 0, OR NOT
  if ((mywarp = get_warp_num ()) != 0) {
    return SUCCESS;
  }

  // First check if a handle for the requested timer already exists (i=1 skips GPTL_ROOT)
  for (i = 1; i <= ntimers; ++i) {
    if (STRMATCH (name, timernames[i].name)) {
#ifdef DEBUG_PRINT
      printf ("%s name=%s: Returning already existing handle=%d\n", thisfunc, name, i);
#endif
      *handle = i;
      return SUCCESS;
    }
  }
  
  if (ntimers >= maxtimers) {
    return GPTLerror_2s1d ("%s: Too many timers. name=%s maxtimers needs to be incremented from %d\n",
			   thisfunc, name, maxtimers);
  } else {
    numchars = MIN (my_strlen (name), MAX_CHARS);
    max_name_len = MAX (numchars, max_name_len);
    *handle = ++ntimers;
    memcpy (timernames[ntimers].name, name, numchars);
    timernames[ntimers].name[numchars] = '\0';
  }
  //  printf ("%s name=%s: mywarp=%d Returning new handle=%d\n", thisfunc, name, mywarp, *handle);
      
  return SUCCESS;
}

/*
** GPTLstart_gpu: start a timer based on a handle
**
** Input arguments:
**   name: timer name (required when on input, handle=0)
**   handle: pointer to timer matching "name"
**
** Return value: 0 (success) or GPTLerror (failure)
*/
__device__ int GPTLstart_gpu (const int handle)
{
  Timer *ptr;        // linked list pointer
  int w;             // warp index (of this thread)
  int wi;            // flattened 2d index for warp number and timer name

  static const char *thisfunc = "GPTLstart_gpu";

#ifdef ENABLE_GPUCHECKS
  if ( ! initialized)
    return GPTLerror_1s1d ("%s handle=%d: GPTLinitialize_gpu has not been called\n", 
			   thisfunc, handle);
#endif
  w = get_warp_num ();

  // Return if not thread 0 of the warp, or warpId is outside range of available timers
  if (w == NOT_ROOT_OF_WARP || w == WARPID_GT_MAXWARPS)
    return SUCCESS;

#ifdef VERBOSE
  printf ("Entered %s w=%d handle=%d\n", thisfunc, w, handle);
#endif

  // Input handle should be a positive integer not greater than ntimers (0 accepted for GPTL_ROOT)
#ifdef ENABLE_GPUCHECKS
  if (handle < 0 || handle > ntimers)
    return GPTLerror_1s1d ("%s: Invalid input handle=%d. Perhaps GPTLinit_handle_gpu not called?\n",
			   thisfunc, handle);
#endif
  wi = FLATTEN_TIMERS (w, handle);
  ptr = &timers[wi];
  
  /* 
  ** Recursion => increment depth in recursion and return.  We need to return 
  ** because we don't want to restart the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */
#ifdef ENABLE_GPUCHECKS
  if (ptr->onflg) {
    ++ptr->recurselvl;
    return SUCCESS;
  }
#endif

#ifdef DEBUG_PRINT
  printf ("%s: ptr=%p setting onflg=true\n", thisfunc, ptr);
#endif

  // Get the timestamp and smid
  // IMPORTANT: Issue the cmds in sequence because SM changing between clock64() call and getting
  // the SM number would be bad.
  // NOTE: Timing value will be thrown away if SM changes upon stop() call.
  ptr->wall.last = clock64 ();
  asm volatile ("mov.u32 %0, %smid;" : "=r"(ptr->smid));
  ptr->onflg = true;

  return SUCCESS;
}

/*
** GPTLstop_gpu: stop a timer based on a handle
**
** Input arguments:
**   name: timer name (used only for diagnostics)
**   handle: pointer to timer
**
** Return value: 0 (success) or -1 (failure)
*/
__device__ int GPTLstop_gpu (const int handle)
{
  register long long tp1;    // time stamp
  Timer timer;               // local copy of timers[wi]: gives some speedup vs. global array
  int w;                     // warp number for this process
  int wi;                    // flattened (1-d) index into 2-d array [timer][warp]
  uint smid;                 // SM id
  static const char *thisfunc = "GPTLstop_gpu";

#ifdef ENABLE_GPUCHECKS
  if ( ! initialized)
    return GPTLerror_1s ("%s: GPTLinitialize_gpu has not been called\n", thisfunc);
#endif
  
  w = get_warp_num ();

  // Return if not thread 0 of the warp, or warpId is outside range of available timers
  if (w == NOT_ROOT_OF_WARP || w == WARPID_GT_MAXWARPS)
    return SUCCESS;

#ifdef VERBOSE
  printf ("Entered %s w=%d handle=%d\n", thisfunc, w, handle);
#endif

#ifdef ENABLE_GPUCHECKS
  // Input handle should be a positive integer not greater than ntimers (0 accepted for GPTL_ROOT)
  if (handle < 0 || handle > ntimers)
    return GPTLerror_1s1d ("%s: Invalid input handle=%d. Perhaps GPTLinit_handle_gpu not called?\n",
			   thisfunc, handle);
#endif
  // Get the timestamp and smid
  // IMPORTANT: Issue the cmds in sequence because SM changing between clock64() call and getting
  // the SM number would be bad.
  // NOTE: Timing value will be thrown away if SM changed from start() call.
  tp1 = clock64 ();
  asm ("mov.u32 %0, %smid;" : "=r"(smid));

  wi = FLATTEN_TIMERS (w, handle);
  timer = timers[wi];

#ifdef ENABLE_GPUCHECKS
  if ( ! timer.onflg )
    return GPTLerror_2s ("%s: timer %s was already off.\n", thisfunc, timernames[handle].name);
#endif
  /* 
  ** Recursion => decrement depth in recursion and return.  We need to return
  ** because we don't want to stop the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */
#ifdef ENABLE_GPUCHECKS
  if (timer.recurselvl > 0) {
    --timer.recurselvl;
    ++timer.count;
    timers[wi] = timer;
    return SUCCESS;
  }
#endif
  if (update_stats_gpu (handle, &timer, tp1, w, smid) != 0)
    return GPTLerror_1s ("%s: error from update_stats_gpu\n", thisfunc);
#ifdef DEBUG_PRINT
  printf ("%s: handle=%d count=%d\n", thisfunc, handle, (int) timer.count);
#endif
  timers[wi] = timer;
  
  return SUCCESS;
}

/*
** update_stats_gpu: update stats inside ptr. Called by GPTLstop_gpu, GPTLstop_handle_gpu
**
** Input arguments:
**   ptr: pointer to timer
**   tp1: input time stamp
**   w: warp index
**
** Return value: 0 (success) or GPTLerror (failure)
*/

__device__ static inline int update_stats_gpu (const int handle,
					       Timer *ptr, 
					       const long long tp1, 
					       const int w,
					       const uint smid)
{
  register long long delta;           // time diff from start()
  static const char *thisfunc = "update_stats_gpu";
#ifdef DEBUG_PRINT
  printf ("%s: ptr=%p setting onflg=false\n", thisfunc, ptr);
#endif

  ptr->onflg = false;
  delta = tp1 - ptr->wall.last;
#ifdef ENABLE_GPUCHECKS
  if (smid != ptr->smid) {
    printf ("GPTL %s: name=%s w=%d sm changed from %d to %d: new kernel? \n"
	    "TIMINGS WITH Bad_SM > 0 PROBABLY INACCURATE.\n"
	    "NEGATIVE STOP MINUS START INCIDENTS WILL BE SKIPPED.\n", 
	    thisfunc, timernames[handle].name, w, ptr->smid, smid);
    ++ptr->badsmid_count;
  }
#endif
  if (delta < 0) {
#ifdef PRINTNEG
    bool isSet; 
    // Use critical section so printf from multiple SMs don't get scrambled
    do {
      // If mutex is 0, grab by setting = 1
      // If mutex is 1, it stays 1 and isSet will be false
      isSet = atomicCAS ((int *) &mutex, 0, 1) == 0; 
      if (isSet) {  // critical section starts here
	printf ("GPTL: %s name=%s w=%d WARNING NEGATIVE DELTA ENCOUNTERED: %lld-%lld=%lld=%g seconds: IGNORING\n", 
		thisfunc, timernames[handle].name, w, tp1, ptr->wall.last, delta, delta / (-gpu_hz));
	printf ("Bit pattern old:");
	prbits8 ((uint64_t) ptr->wall.last);

	printf ("Bit pattern new:");
	prbits8 ((uint64_t) tp1);
	mutex = 0;     // end critical section by releasing the mutex
      }
    } while ( !isSet); // exit the loop after critical section executed
#endif
    
    ++ptr->negdelta_count;
    return SUCCESS;
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
  return SUCCESS;
}

/*
** GPTLreset_gpu: reset all timers to 0
**
** Return value: 0 (success) or GPTLerror (failure)
*/
__global__ void GPTLreset_gpu (void)
{
  int i;
  int w;
  int wi;
  int maxwarpid_timed;
  static const char *thisfunc = "GPTLreset_gpu";

  if ( ! initialized) {
    (void) GPTLerror_1s ("%s: GPTLinitialize_gpu has not been called\n", thisfunc);
    return;
  }

  if (get_warp_num () != 0)
    return;

  maxwarpid_timed = GPTLget_maxwarpid_timed ();

  for (w = 0; w <= maxwarpid_timed; ++w) {
    for (i = 0; i < maxtimers; ++i) {
      wi = FLATTEN_TIMERS(w,i);
      timers[wi].onflg = false;
      timers[wi].count = 0;
      memset (&timers[wi].wall, 0, sizeof (timers[wi].wall));
    }
  }

  if (verbose)
    printf ("%s: accumulators for all GPU timers set to zero\n", thisfunc);
}

__device__ static inline int get_warp_num ()
{
  int threadId;
  int warpId;

  threadId = threadIdx.x
        +  blockDim.x  * threadIdx.y
        +  blockDim.x  *  blockDim.y  * threadIdx.z
        +  blockDim.x  *  blockDim.y  *  blockDim.z  * blockIdx.x
        +  blockDim.x  *  blockDim.y  *  blockDim.z  *  gridDim.x  * blockIdx.y
        +  blockDim.x  *  blockDim.y  *  blockDim.z  *  gridDim.x  *  gridDim.y  * blockIdx.z;

  // Only thread 0 of the warp will be timed
  if (threadId % warpsize != 0)
    return NOT_ROOT_OF_WARP;

  warpId = threadId / warpsize;

  // Setting maxwarpid_found is a race condition that is ignored due to efficiency considerations
  // It is only printed as an estimate when GPTLpr is called.
#ifdef ENABLE_FOUND
  if (warpId+1 > maxwarpid_found)
    maxwarpid_found = warpId;
#endif
  
  if (warpId > maxwarps-1)
    return WARPID_GT_MAXWARPS;

  return warpId;
}

__device__ int GPTLget_maxwarpid_timed (void)
{
  int wi;
  int maxwarpid_timed = 0;
  static const char *thisfunc = "GPTLget_maxwarpid_timed";

  if (get_warp_num () != 0) {
    (void) GPTLerror_1s ("%s: must only be called by thread 0 of warp 0\n", thisfunc);
    return -1;
  }

  for (int w = 0; w < maxwarps; ++w) {
    for (int i = ntimers; i > 0; --i) {
      wi = FLATTEN_TIMERS(w,i);
      if (timers[wi].count > 0 && w > maxwarpid_timed)
	maxwarpid_timed = w;
    }
  }
  return maxwarpid_timed;
}

__device__ int GPTLget_wallclock_gpu (const int handle,
				      double *accum, double *max, double *min)
{
  int w;
  int wi;
  static const char *thisfunc = "GPTLget_wallclock_gpu";
  
  if ( ! initialized)
    (void) GPTLerror_1s ("%s: GPTLinitialize_gpu has not been called\n", thisfunc);

  if (gpu_hz == 0.)
    (void) GPTLerror_1s ("%s: gpu_hz has not been set\n", thisfunc);

  w = get_warp_num ();
  if (w == NOT_ROOT_OF_WARP || w == WARPID_GT_MAXWARPS)
    return SUCCESS;

  if (handle < 0 || handle > ntimers)
    return GPTLerror_1s1d ("%s: bad handle=%d\n", thisfunc, handle);

  wi = FLATTEN_TIMERS (w, handle);
  
  *accum = timers[wi].wall.accum / gpu_hz;
  *max   = timers[wi].wall.max   / gpu_hz;
  *min   = timers[wi].wall.min   / gpu_hz;
  return 0;
}

//JR want to use variables to dimension arrays but nvcc is not C99 compliant
__global__ void GPTLfill_gpustats (Gpustats *gpustats, 
				   int *max_name_len_out,
				   int *ngputimers)
{
  int w;
  int n;
  int maxwarpid_timed;
  static const char *thisfunc = "GPTLfill_gpustats";

  if ( ! initialized) {
    (void) GPTLerror_1s ("%s: GPTLinitialize_gpu has not been called\n", thisfunc);
    return;
  }

  if (get_warp_num () != 0) {
    (void) GPTLerror_1s ("%s: must only be called by thread 0 of warp 0\n", thisfunc);
    return;
  }

  maxwarpid_timed = GPTLget_maxwarpid_timed ();
  *max_name_len_out = max_name_len;
  *ngputimers = ntimers;

  // Step 1: process entries for all warps based on those in warp 0
  // gpustats starts at 0. timers start at 1
  for (n = 0; n < ntimers; ++n) {
    init_gpustats (&gpustats[n], n+1);
    for (w = 1; w <= maxwarpid_timed; ++w) {
      fill_gpustats (&gpustats[n], n+1, w);
    }
  }

#ifdef DEBUG_PRINT
  printf ("%s: ngputimers=%d\n", thisfunc, n);
  for (n = 0; n < *ngputimers; ++n) {
    printf ("%s: timer=%s accum_max=%lld accum_min=%lld count_max=%d nwarps=%d\n", 
	    thisfunc, gpustats[n].name, gpustats[n].accum_max, gpustats[n].accum_min, gpustats[n].count_max, gpustats[n].nwarps);
  }
#endif
  return;
}

__device__ static void init_gpustats (Gpustats *gpustats, int idx)
{
  const int w = 0;
  (void) my_strcpy (gpustats->name, timernames[idx].name);
  gpustats->count  = timers[idx].count;
  if (timers[idx].count > 0)
    gpustats->nwarps = 1;
  else
    gpustats->nwarps = 0;

  gpustats->accum_max      = timers[idx].wall.accum;
  gpustats->accum_max_warp = w;

  gpustats->accum_min      = timers[idx].wall.accum;
  gpustats->accum_min_warp = w;

  gpustats->count_max      = timers[idx].count;
  gpustats->count_max_warp = w;

  gpustats->count_min      = timers[idx].count;
  gpustats->count_min_warp = w;

  gpustats->negdelta_count_max       = timers[idx].negdelta_count;
  gpustats->negdelta_count_max_warp  = w;
  gpustats->negdelta_nwarps          = timers[idx].negdelta_count  > 0 ? 1 : 0;

  gpustats->badsmid_count  = timers[idx].badsmid_count;
}

__device__ static void fill_gpustats (Gpustats *gpustats, int idx, int w)
{
  int wi = FLATTEN_TIMERS (w,idx);
  
  if (timers[wi].count > 0) {
    gpustats->count += timers[wi].count;
    ++gpustats->nwarps;

    if (timers[wi].wall.accum > gpustats->accum_max) {
      gpustats->accum_max      = timers[wi].wall.accum;
      gpustats->accum_max_warp = w;
    }
    
    if (timers[wi].wall.accum < gpustats->accum_min) {
      gpustats->accum_min      = timers[wi].wall.accum;
      gpustats->accum_min_warp = w;
    }
    
    if (timers[wi].count > gpustats->count_max) {
      gpustats->count_max      = timers[wi].count;
      gpustats->count_max_warp = w;
    }
    
    if (timers[wi].count < gpustats->count_min) {
      gpustats->count_min      = timers[wi].count;
      gpustats->count_min_warp = w;
    }
    
    if (timers[wi].negdelta_count > gpustats->negdelta_count_max) {
      gpustats->negdelta_count_max      = timers[wi].negdelta_count;
      gpustats->negdelta_count_max_warp = w;
    }

    if (timers[wi].negdelta_count > 0)
      ++gpustats->negdelta_nwarps;

    gpustats->badsmid_count += timers[wi].badsmid_count;
  }
}

__device__ static int my_strlen (const char *str)
{
  const char *s;
  for (s = str; *s; ++s);
  return(s - str);
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
__device__ static int my_strcmp (const char *str1, const char *str2)
{
#ifndef MINE
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
**   utr_ohd:            Underlying timer routine
**   self_ohd:           Estimate of GPTL-induced overhead in the timer itself (included in "Wallclock")
**   parent_ohd:         Estimate of GPTL-induced overhead for the timer which appears in its parents
*/
__global__ void GPTLget_overhead_gpu (int *maxwarpid_timed_out,
				      int *maxwarpid_found_out,
				      long long *get_warp_num_ohd,  // Getting my warp index
				      long long *startstop_ohd,     // start/stop pair
				      long long *utr_ohd,           // Underlying timing routine
				      long long *start_misc_ohd,    // misc start code
				      long long *stop_misc_ohd,     // misc stop code
				      long long *self_ohd,          // OHD in timer itself
				      long long *parent_ohd,        // OHD in parent
				      long long *my_strlen_ohd,
				      long long *STRMATCH_ohd)
{
  volatile uint smid;         // SM id
  long long t1, t2;           // Initial, final timer values
  int i;
  int ret;
  int mywarp;                 // our warp number
  char name[MAX_CHARS+1];     // Name to be used for various OHD tests
  char samename[MAX_CHARS+1]; // Copy of "name" for STRMATCH test

  *maxwarpid_timed_out = GPTLget_maxwarpid_timed ();
  *maxwarpid_found_out = maxwarpid_found;
  
  // Define name to be used in OHD estimates. Use GPTL_ROOT because it's always there
  my_strcpy (name, timernames[0].name); // GPTL_ROOT
  my_strcpy (samename, name);

  /*
  ** Gather timings by running each test 1000 times
  ** First: start/stop overhead 
  */
  t1 = clock64();
  for (i = 0; i < 1000; ++i) {
    ret = GPTLstart_gpu (0);
    ret = GPTLstop_gpu (0);
  }
  t2 = clock64();
  startstop_ohd[0] = (t2 - t1) / 1000;

  // get_warp_num overhead. Need a bogus computation or compiler may optimize out the code
  t1 = clock64();
  for (i = 0; i < 1000; ++i) {
    if ((mywarp = get_warp_num ()) < -999)
      get_warp_num_ohd[0] = -999;
  }
  t2 = clock64();
  get_warp_num_ohd[0] = (t2 - t1) / 1000;

  // utr overhead
  t1 = clock64();
  for (i = 0; i < 1000; ++i) {
    asm volatile ("mov.u32 %0, %smid;" : "=r"(smid));
    t2 = clock64();
  }
  *utr_ohd = (t2 - t1) / 1000;

  // start misc overhead
  t1 = clock64();
  for (i = 0; i < 1000; ++i) {
    start_misc (0, 0);  // w, handle (handle=0 is GPTL_ROOT)
  }
  t2 = clock64();
  start_misc_ohd[0] = (t2 - t1) / 1000;

  // stop misc overhead
  t1 = clock64();
  for (i = 0; i < 1000; ++i) {
    stop_misc (0, 0);  // w, handle (handle=0 is GPTL_ROOT)
  }
  t2 = clock64();
  stop_misc_ohd[0] = (t2 - t1) / 1000;

  // Self and parent OHD estimates: A few settings at the end of GPTLstart_gpu should instead be 
  // applied to parent. A few settings at the beginning of GPTLstop_gpu should instead be
  // applied to self. But those errors are likely minor.
  self_ohd[0]   = utr_ohd[0] + start_misc_ohd[0];
  parent_ohd[0] = utr_ohd[0] + 2*get_warp_num_ohd[0] + stop_misc_ohd[0];

  // my_strlen overhead
  t1 = clock64();
  for (i = 0; i < 1000; ++i) {
    ret = my_strlen (name);
  }
  t2 = clock64();
  *my_strlen_ohd = (t2 - t1) / 1000;

  // STRMATCH overhead
  t1 = clock64();
  for (i = 0; i < 1000; ++i) {
    ret = STRMATCH (samename, name);
  }
  t2 = clock64();
  *STRMATCH_ohd = (t2 - t1) / 1000;
  return;
}

__device__ static void start_misc (int w, const int handle)
{
  int wi;
  Timer *ptr;
  static const char *thisfunc = "startmisc";

#ifdef ENABLE_GPUCHECKS
  if ( ! initialized)
    printf ("%s: ! initialized\n", thisfunc);
#endif
  if (w == NOT_ROOT_OF_WARP || w == WARPID_GT_MAXWARPS)
    printf ("%s: bad w value\n", thisfunc);

#ifdef ENABLE_GPUCHECKS
  if (handle < 0 || handle > ntimers)
    printf ("%s: bad handle value %d\n", thisfunc, handle);
#endif
  wi = FLATTEN_TIMERS (w, handle);
  ptr = &timers[wi];

#ifdef ENABLE_GPUCHECKS
  if (ptr->onflg) {
    ++ptr->recurselvl;
    printf ("%s: onflg should be off\n", thisfunc);
    ptr->smid = 0;
    ptr->wall.last = 0L;
  }
#endif
  ptr->onflg = false;  // GPTLstart actually sets this true but set false for better OHD est.
}

__device__ static void stop_misc (int w, const int handle)
{
  int wi;
  Timer timer;
  static const char *thisfunc = "stopmisc";

#ifdef ENABLE_GPUCHECKS
  if ( ! initialized)
    printf ("%s: ! initialized\n", thisfunc);
  if (w == NOT_ROOT_OF_WARP || w == WARPID_GT_MAXWARPS)
    printf ("%s: bad w value\n", thisfunc);

  if (handle < 0 || handle > ntimers)
    printf ("%s: bad handle value %d\n", thisfunc, handle);
#endif

  wi = FLATTEN_TIMERS (w, handle);
  timer = timers[wi];

#ifdef ENABLE_GPUCHECKS
  if ( timer.onflg )
    printf ("%s: onflg was on\n", thisfunc); // Invert logic for better OHD est.
  if (timer.recurselvl > 0) {
    --timer.recurselvl;
    ++timer.count;
  }
#endif

  // Last 3 args are timestamp, w, smid
  if (update_stats_gpu (handle, &timer, timer.wall.last, 0, 0) != 0)
    printf ("%s: problem with update_stats_gpu\n", thisfunc);
  timers[wi] = timer;
}

__global__ void GPTLget_memstats_gpu (float *regionmem, float *timernamemem)
{
  *regionmem    = (float) maxwarps * maxtimers * sizeof (Timer);
  *timernamemem = (float)            maxtimers * sizeof (Timername);
  return;
}

__device__ int GPTLmy_sleep (float seconds)
{
  volatile long long start, now;
  volatile double delta;
  static const char *thisfunc = "GPTLmy_sleep";

  int mywarp = get_warp_num ();

  // Only sleep if we're root of warp
  if (mywarp == NOT_ROOT_OF_WARP)
    return SUCCESS;

  if (gpu_hz == 0.)
    return GPTLerror_1s ("%s: need to set gpu_hz via call to GPTLinitialize_gpu() first\n",
			 thisfunc);

  start = clock64();
  do {
    now = clock64();
    delta = (now - start) / gpu_hz;
  } while (delta < seconds);

  return SUCCESS;
}

__device__ void GPTLdummy_gpu ()
{
  return;
}

#ifdef PRINTNEG
__device__ static void prbits8 (uint64_t val)
{
  uint64_t mask = 1;
  char chars[64];
  
  int i;

  for (i = 0; i < 64; ++i) {
    if ((val & mask) == 0) 
      chars[i] = '0';
    else
      chars[i] = '1';
    val >>= 1;
  }
  
  for (i = 0; i < 64; ++i) {
    printf ("%c", chars[63-i]);
    if ((i+1) % 8 == 0)
      printf (" ");
  }
  printf ("\n");
}
#endif
  
__device__ int GPTLget_warp_thread (int *warp, int *thread)
{
  static const char *thisfunc = "GPTLget_warp_thread";
  if ( ! initialized) {
    (void) GPTLerror_1s ("%s: initialization was not completed\n", thisfunc);
    return -1;
  }

  *thread = threadIdx.x
        +  blockDim.x  * threadIdx.y
        +  blockDim.x  *  blockDim.y  * threadIdx.z
        +  blockDim.x  *  blockDim.y  *  blockDim.z  * blockIdx.x
        +  blockDim.x  *  blockDim.y  *  blockDim.z  *  gridDim.x  * blockIdx.y
        +  blockDim.x  *  blockDim.y  *  blockDim.z  *  gridDim.x  *  gridDim.y  * blockIdx.z;
  *warp = (*thread) / warpsize;
  return 0;
}

__device__ int GPTLsliced_up_how (const char *txt)
{
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
      blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    printf ("GPTLsliced_up_how: %s\n", txt);
    if (blockDim.x > 1)
      printf ("blockDim.x=%d ", blockDim.x);
    if (blockDim.y > 1)
      printf ("blockDim.y=%d ", blockDim.y);
    if (blockDim.z > 1)
      printf ("blockDim.z=%d ", blockDim.z);
    printf ("\n");

    if (gridDim.x > 1)
      printf ("gridDim.x=%d ", gridDim.x);
    if (gridDim.y > 1)
      printf ("gridDim.y=%d ", gridDim.y);
    printf ("\n");
  }
  return 0;
}
}
