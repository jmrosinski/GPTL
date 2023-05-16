/*
** gptl.cu
** Author: Jim Rosinski
**
** Main file contains most CUDA GPTL functions
*/

#include "config.h"        // Must be first include.
// gptl gpu-private 
#include "gptl_cuda.h"
#include "api.h"
#include "stringfuncs.h"
#include "util.h"
#include "timingohd.h"
// system
#include <stdio.h>
#include <string.h>        // memcpy
#include <stdint.h>        // uint types
#include <cuda.h>

namespace api {
  __device__ {
    Timer *timers = 0;            // array (also linked list) of timers
    Timername *timernames;        // array of timer names
    int max_name_len;             // max length of timer name
    int ntimers = 0;              // number of timers
    int maxwarpid_found = 0;      // number of warps found : init to 0 
    bool verbose = false;         // output verbosity                  
    double gpu_hz = 0.;           // clock freq                        
    int warps_per_sm = 0;         // used for overhead calcs         
#ifdef ENABLE_CONSTANTMEM
    __constant__ bool initialized = false; // GPTLinitialize has been called
    __constant__ int maxtimers = 0;   // max number of timers allowed
    __constant__ int warpsize = 0;    // warp size
    __constant__ int maxwarps = 0;    // max number of warps that will be examined
#else
                 bool initialized = false;     // GPTLinitialize has been called
		 int maxtimers = 0;   // max number of timers allowed
		 int warpsize = 0;    // warp size
		 int maxwarps = 0;    // max number of warps that will be examined
#endif
  }
}

#ifdef TIME_GPTL
namespace timingohd {
  __device__ {
    const char *internal_name[NUM_INTERNAL_TIMERS] = {"GPTLstart_gpu",
						      "GPTLstop_gpu",
						      "update_stats"};
    const long long *globcount = 0;            // for timing GPTL itself
    // Indices for internal timers
    const int istart = 0;
    const int istop = 1;
    const int update_stats = 2;
  }
}
#endif

// Defining PRINTNEG will print to stdout whenever a negative interval (stop minus start) is
// encountered. Only useful when non-zero negative intervals are reported in timing output
// Should be turned OFF normally--very expensive even when no negatives found.
#undef PRINTNEG
#ifdef PRINTNEG
__device__ static volatile int mutex_print = 0; // critical section unscrambles printf output
__device__ static void prbits8 (uint64_t);
#endif

// VERBOSE is a debugging ifdef local to the rest of this file
#undef VERBOSE

// All user-visible routines must be C-callable to enable call-from-C and call-from-Fortran
extern "C" {

/*
** GPTLinit_handle_gpu: Initialize a handle for further use by GPTLstart_gpu() and GPTLstop_gpu()
**
** Input arguments:
**   name: timer name
**
** Output arguments:
**   handle: Index into array for "name"
**
** Return value: 0 (success) or util::error* (failure)
*/
__device__ int GPTLinit_handle_gpu (const char *name, int *handle)
{
  int numchars;      // length of "name"
  int mywarp;        // my warp number
  int i;
  static const char *thisfunc = "GPTLinit_handle_gpu";

  // Initialize handle to a bad value. This prevents mistakes e.g. "acc copyout(handle)" which
  // MAY end up setting handle=0 if this routine fails (valid but not desired GPTL_ROOT).
  // Note "acc copy(handle)" for this routine is better
  *handle = -999;

  if ( ! api::initialized)
    return util::error_1s ("%s: GPTLinitialize has not been called\n", thisfunc);
  
  // Guts of this function are run only by thread 0 of warp 0 to prevent race conditions on handle. 
  // Nice feature: Can be called by just thread 0 of warp 0, OR NOT
  if ((mywarp = api::get_warp_num ()) != 0)
    return util::error_1s1d ("%s: must be called ONLY from thread 0 warp 0 got warp=%d\n",
			     thisfunc, mywarp);

  // Check if a handle for the requested timer already exists (i=1 skips GPTL_ROOT)
  for (i = 1; i <= api::ntimers; ++i) {
    if (STRMATCH (name, api::timernames[i].name)) {
#ifdef DEBUG_PRINT
      printf ("%s name=%s: Returning already existing handle=%d\n", thisfunc, name, i);
#endif
      *handle = i;
      return SUCCESS;
    }
  }
  
  if (api::ntimers >= api::maxtimers-1) {
    return util::error_2s1d ("%s: Too many timers. name=%s maxtimers %d is too small\n",
			     thisfunc, name, api::maxtimers);
  }
  // End of error checks. Initialize the handle

  numchars = MIN (my_strlen (name), MAX_CHARS);
  api::max_name_len = MAX (numchars, api::max_name_len);
  *handle = ++api::ntimers;
  memcpy (api::timernames[api::ntimers].name, name, numchars);
  api::timernames[api::ntimers].name[numchars] = '\0';
  //  printf ("%s name=%s: mywarp=%d Returning new handle=%d\n", thisfunc, name, mywarp, *handle);
  return SUCCESS;
}

/*
** GPTLstart_gpu: start a timer based on a handle
**
** Input arguments:
**   handle: Index of timer to start
**
** Return value: 0 (success) or util::error (failure)
*/
__device__ int GPTLstart_gpu (const int handle)
{
  Timer *ptr;        // linked list pointer
  int w;             // warp index (of this thread)
  int wi;            // flattened 2d index for warp number and timer name
  static const char *thisfunc = "GPTLstart_gpu";

#ifdef DUMMYGPUSTARTSTOP
  return SUCCESS;
#endif

#ifdef TIME_GPTL
  long long start = clock64 ();  // starting timestamp
#endif

#ifdef ENABLE_GPUCHECKS
  if ( ! api::initialized)
    return util::error_1s1d ("%s handle=%d: GPTLinitialize has not been called\n", 
			     thisfunc, handle);
#endif
  w = api::get_warp_num ();

  // Return if not thread 0 of the warp, or warpId is outside range of available timers
  if (w < 0)
    return SUCCESS;

#ifdef VERBOSE
  printf ("Entered %s w=%d handle=%d\n", thisfunc, w, handle);
#endif

  // Input handle should be a positive integer not greater than ntimers (GPTL_ROOT is 0))
#ifdef ENABLE_GPUCHECKS
  if (handle < 0 || handle > api::ntimers)
    return util::error_1s1d ("%s: Invalid input handle=%d. GPTLinit_handle_gpu not called?\n",
			   thisfunc, handle);
#endif
  wi = FLATTEN_TIMERS (w, handle);
  ptr = &api::timers[wi];
  
#ifdef ENABLE_GPURECURSION
  /* 
  ** Recursion => increment depth in recursion and return.  We need to return 
  ** because we don't want to restart the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */
  if (ptr->onflg) {
    ++ptr->recurselvl;
    return SUCCESS;
  }
#endif

#ifdef DEBUG_PRINT
  printf ("%s: ptr=%p setting onflg=true\n", thisfunc, ptr);
#endif

  // Get the timestamp and smid.
  asm volatile ("mov.u32 %0, %smid;" : "=r"(ptr->smid));
  ptr->wall.last = clock64 ();
  ptr->onflg = true;
#ifdef TIME_GPTL
  globcount[istart*api::maxwarps + w] += ptr->wall.last - start;
#endif
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
  Timer timer;               // local copy of api::timers[wi]: gives some speedup vs. global array
  int w;                     // warp number for this process
  int wi;                    // flattened (1-d) index into 2-d array [timer][warp]
  uint smid;                 // SM id
  static const char *thisfunc = "GPTLstop_gpu";

#ifdef DUMMYGPUSTARTSTOP
  return SUCCESS;
#endif

#ifdef TIME_GPTL
  tp1 = clock64 ();
#endif

#ifdef ENABLE_GPUCHECKS
  if ( ! api::initialized)
    return util::error_1s1d ("%s handle=%d: GPTLinitialize has not been called\n", 
			     thisfunc, handle);
#endif
  
  w = api::get_warp_num ();

  // Return if not thread 0 of the warp, or warpId is outside range of available timers
  if (w < 0)
    return SUCCESS;

#ifdef VERBOSE
  printf ("Entered %s w=%d handle=%d\n", thisfunc, w, handle);
#endif

#ifdef ENABLE_GPUCHECKS
  // Input handle should be a positive integer not greater than ntimers (0 accepted for GPTL_ROOT)
  if (handle < 0 || handle > api::ntimers)
    return util::error_1s1d ("%s: Invalid input handle=%d. GPTLinit_handle_gpu not called?\n",
			   thisfunc, handle);
#endif
  // Get the timestamp and smid
#ifndef TIME_GPTL
  tp1 = clock64 ();
#endif
  asm ("mov.u32 %0, %smid;" : "=r"(smid));

  wi = FLATTEN_TIMERS (w, handle);
  timer = api::timers[wi];

#ifdef ENABLE_GPUCHECKS
  if ( ! timer.onflg )
    return util::error_2s ("%s: timer %s was already off.\n",
			   thisfunc, api::timernames[handle].name);
#endif
  /* 
  ** Recursion => decrement depth in recursion and return.  We need to return
  ** because we don't want to stop the timer.  We want the reported time for
  ** the timer to reflect the outermost layer of recursion.
  */
#ifdef ENABLE_GPURECURSION
  if (timer.recurselvl > 0) {
    --timer.recurselvl;
    ++timer.count;
    api::timers[wi] = timer;
    return SUCCESS;
  }
#endif

#ifdef TIME_GPTL
  long long start = clock64 ();
#endif

  api::update_stats (handle, &timer, tp1, w, smid);

#ifdef DEBUG_PRINT
  printf ("%s: handle=%d count=%d\n", thisfunc, handle, (int) timer.count);
#endif
  api::timers[wi] = timer;
  
#ifdef TIME_GPTL
  long long stop = clock64 ();
  globcount[istop*api::maxwarps + w]        += stop - tp1;
  globcount[update_stats*api::maxwarps + w] += stop - start;
#endif
  
  return SUCCESS;
}

__device__ int GPTLget_wallclock_gpu (const int handle, double *accum, double *max, double *min)
{
  int w, wi;
  static const char *thisfunc = "GPTLget_wallclock_gpu";
  
  if ( ! api::initialized)
    return util::error_1s ("%s: GPTLinitialize_gpu has not been called\n", thisfunc);

  if (api::gpu_hz == 0.)
    return util::error_1s ("%s: gpu_hz has not been set\n", thisfunc);

  w = api::get_warp_num ();
  if (w < 0)
    return SUCCESS;

  if (handle < 0 || handle > api::ntimers)
    return util::error_1s1d ("%s: bad handle=%d\n", thisfunc, handle);

  wi = FLATTEN_TIMERS (w, handle);
  
  *accum = api::timers[wi].wall.accum / api::gpu_hz;
  *max   = api::timers[wi].wall.max   / api::gpu_hz;
  *min   = api::timers[wi].wall.min   / api::gpu_hz;
  return 0;
}

__device__ int GPTLmy_sleep (float seconds)
{
  volatile long long start, now;
  volatile double delta;
  static const char *thisfunc = "GPTLmy_sleep";

  int mywarp = api::get_warp_num ();

  // Only sleep if we're root of warp
  if (mywarp == NOT_ROOT_OF_WARP)
    return SUCCESS;

  if (api::gpu_hz == 0.)
    return util::error_1s ("%s: need to set gpu_hz via call to GPTLinitialize_gpu() first\n",
			 thisfunc);

  start = clock64();
  do {
    now = clock64();
    delta = (now - start) / api::gpu_hz;
  } while (delta < seconds);

  return SUCCESS;
}

__device__ void GPTLdummy_gpu ()
{
  return;
}

__device__ int GPTLget_warp_thread (int *warp, int *thread)
{
  *thread = threadIdx.x
        +  blockDim.x  * threadIdx.y
        +  blockDim.x  *  blockDim.y  * threadIdx.z
        +  blockDim.x  *  blockDim.y  *  blockDim.z  * blockIdx.x
        +  blockDim.x  *  blockDim.y  *  blockDim.z  *  gridDim.x  * blockIdx.y
        +  blockDim.x  *  blockDim.y  *  blockDim.z  *  gridDim.x  *  gridDim.y  * blockIdx.z;
  *warp = (*thread) / api::warpsize;
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

__device__ int GPTLget_sm_thiswarp (int smarr[])
{
  int mywarp;
  uint smid;
  
  mywarp = api::get_warp_num ();
  if (mywarp < 0)
    return -1;
  
  asm volatile ("mov.u32 %0, %smid;" : "=r"(smid));
  smarr[mywarp] = smid;
  return mywarp;
}
////////////////////////////////////////////////////////////////////////////////////////////
// End of user-callable region. Name-mangling resumes here
////////////////////////////////////////////////////////////////////////////////////////////
}
  
/*
** update_stats: update stats inside ptr. Called by GPTLstop_gpu
**
** Arguments:
**   handle: index generated by a previous call GPTLinit_handle_gpu
**   ptr: pointer to timer.
**   tp1: input time stamp
**   w: warp index
**   smid: input SM index
*/
__device__ inline void api::update_stats (const int handle,
					  Timer *ptr, 
					  const long long tp1, 
					  const int w,
					  const uint smid)
{
  register long long delta;           // time diff from start()
  static const char *thisfunc = "update_stats";

#ifdef DEBUG_PRINT
  printf ("%s: ptr=%p setting onflg=false\n", thisfunc, ptr);
#endif

  ptr->onflg = false;
  delta = tp1 - ptr->wall.last;
#ifdef ENABLE_GPUCHECKS
  if (smid != ptr->smid) {
    (void) util::error_2s3d ("%s: name=%s w=%d sm changed from %d to %d between GPTLstart_gpu and "
			     "GPTL_stop_gpu\n"
			     "TIMINGS WITH SM CHANGED BETWEEN START AND STOP PROBABLY INACCURATE.\n"
			     "NEGATIVE STOP MINUS START INCIDENTS WILL BE SKIPPED.\n",
			     thisfunc, api::timernames[handle].name, w, ptr->smid, smid);
    ++ptr->badsmid_count;
  }
#endif
  if (delta < 0) {
#ifdef PRINTNEG
    get_mutex (&mutex_print);
    printf ("GPTL: %s name=%s w=%d WARNING NEGATIVE DELTA ENCOUNTERED: %lld-%lld=%lld=%g seconds:"
	    "IGNORING\n", thisfunc, api::timernames[handle].name, w, tp1, ptr->wall.last, delta,
	    delta / (-api::gpu_hz));
    printf ("Bit pattern old:");
    prbits8 ((uint64_t) ptr->wall.last);

    printf ("Bit pattern new:");
    prbits8 ((uint64_t) tp1);
    free_mutex (&mutex_print);
#endif
    
    ++ptr->negdelta_count;

  } else {

    ++ptr->count;
    ptr->wall.accum += delta;
  
    if (delta > ptr->wall.max)  // On first call ptr->wall.max will be 0
      ptr->wall.max = delta;
    if (delta < ptr->wall.min)  // On first call ptr->wall.min will be LLONG_MAX
      ptr->wall.min = delta;
  }
  return;
}

/*
** api::get_warp_num: flatten CUDA thread, block, grid location to a linearized warp number
**
** Return value: warp number, or a negative number if not root of warp, or outside bounds
** NOTE: Code structure gives 100X speedup vs. multiply nested "if"s
*/
__device__ inline int api::get_warp_num ()
{
  register int threadId;
  register int warpId;
  register int retval;

  threadId = threadIdx.x
        +  blockDim.x  * threadIdx.y
        +  blockDim.x  *  blockDim.y  * threadIdx.z
        +  blockDim.x  *  blockDim.y  *  blockDim.z  * blockIdx.x
        +  blockDim.x  *  blockDim.y  *  blockDim.z  *  gridDim.x  * blockIdx.y
        +  blockDim.x  *  blockDim.y  *  blockDim.z  *  gridDim.x  *  gridDim.y  * blockIdx.z;

  warpId = threadId / api::warpsize;

  // Setting maxwarpid_found is a race condition that is ignored due to efficiency considerations
  // It is only printed as an estimate when GPTLpr is called.
#ifdef ENABLE_FOUND
  if (warpId+1 > api::maxwarpid_found)
    api::maxwarpid_found = warpId;
#endif

  retval = warpId;
  if (threadId % api::warpsize != 0)
    retval = NOT_ROOT_OF_WARP;
  else if (warpId > api::maxwarps-1)
    retval = WARPID_GT_MAXWARPS;
  
  // linearized warp number, or negative number which caller will handle appropriately
  return retval;
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
