#include "config.h" // Must be first include.
// gptl gpu-private 
#include "overhead.h"
#include "devicehost.h"
#include "api.h"
#include "stringfuncs.h"
#include "gptl_cuda.h"
// system
#include <stdio.h>

__device__ static void start_misc (int, const int);
__device__ static void stop_misc (int w, const int handle);

/*
** get_overhead_gpu: return current status info about a timer. If certain stats are not enabled, 
** they should just have zeros in them.
** 
** Output args:
**   get_warp_num_ohd: Getting my warp index
**   utr_ohd:          Underlying timer routine
**   self_ohd:         Estimate of GPTL-induced overhead in the timer itself (included in "Wallclock")
**   parent_ohd:       Estimate of GPTL-induced overhead for the timer which appears in its parents
*/
__global__ void overhead::get_overhead_gpu (float *get_warp_num_ohd,  // Getting my warp index
					    float *startstop_ohd,     // start/stop pair
					    float *utr_ohd,           // Underlying timing routine
					    float *start_misc_ohd,    // misc start code
					    float *stop_misc_ohd,     // misc stop code
					    float *self_ohd,          // OHD in timer itself
					    float *parent_ohd,        // OHD in parent
					    float *my_strlen_ohd,     // strlen overhead
					    float *STRMATCH_ohd,      // STRMATCH overhead
					    int *retval)              // return value
{
  volatile uint smid;         // SM id
  long long t1, t2;           // Initial, final timer values
  int i;
  int ret;
  int mywarp;                 // our warp number
  char name[MAX_CHARS+1];     // Name to be used for various OHD tests
  char samename[MAX_CHARS+1]; // Copy of "name" for STRMATCH test
  static const int iters = 1000;
  static const char *thisfunc = "get_overhead_gpu";

  // Define name to be used in OHD estimates. Use GPTL_ROOT because it's always there
  stringfuncs::my_strcpy (name, api::timernames[0].name); // GPTL_ROOT
  stringfuncs::my_strcpy (samename, name);

  // Gather timings by running each test "iters" times. We are running on a single SM, with all
  // cores active. Mimic the use by GPTLstart_gpu and GPTLstop_gpu by only thread 0 does work

  // Return if not thread 0 of the warp
  if ((mywarp = GPTLget_warp_num ()) < 0)
    return;
  
  if (mywarp > api::warps_per_sm - 1) {
    *retval = -1;
    printf ("%s: mywarp=%d must be < warps_per_sm=%d. No GPU overhead stats will be gathered\n",
	    thisfunc, mywarp, api::warps_per_sm);
    return;
  }

  // First: start/stop overhead. First 2 are warmups. "0" index is GPTL_ROOT
  ret = GPTLstart_gpu (0);
  ret = GPTLstop_gpu (0);
  t1 = clock64();
  for (i = 0; i < iters; ++i) {
    ret = GPTLstart_gpu (0);
    ret = GPTLstop_gpu (0);
  }
  t2 = clock64();
  startstop_ohd[mywarp] = (t2 - t1) / (float) iters;

  // get_warp_num overhead. Need a bogus computation or compiler may optimize out the code
  t1 = clock64();
  for (i = 0; i < iters; ++i) {
    if ((GPTLget_warp_num ()) < -999)
      get_warp_num_ohd[mywarp] = -999;  // Will actually NEVER get set
  }
  t2 = clock64();
  get_warp_num_ohd[mywarp] = (t2 - t1) / (float) iters;

  // utr plus smid overhead
  t1 = clock64();
  for (i = 0; i < iters; ++i) {
    asm volatile ("mov.u32 %0, %smid;" : "=r"(smid));
    t2 = clock64();
  }
  utr_ohd[mywarp] = (t2 - t1) / (float) iters;

  // start misc overhead
  t1 = clock64();
  for (i = 0; i < iters; ++i) {
    start_misc (mywarp, 0);  // w, handle (handle=0 is GPTL_ROOT)
  }
  t2 = clock64();
  start_misc_ohd[mywarp] = (t2 - t1) / (float) iters;

  // stop misc overhead
  t1 = clock64();
  for (i = 0; i < iters; ++i) {
    stop_misc (mywarp, 0);  // w, handle (handle=0 is GPTL_ROOT)
  }
  t2 = clock64();
  stop_misc_ohd[mywarp] = (t2 - t1) / (float) iters;

  // Self and parent OHD estimates: A few settings at the end of GPTLstart_gpu should instead be 
  // applied to parent. A few settings at the beginning of GPTLstop_gpu should instead be
  // applied to self. But those errors are likely minor.
  self_ohd[mywarp]   = utr_ohd[mywarp] + start_misc_ohd[mywarp];
  parent_ohd[mywarp] = utr_ohd[mywarp] + 2*get_warp_num_ohd[mywarp] + stop_misc_ohd[mywarp];

  // my_strlen overhead
  t1 = clock64();
  for (i = 0; i < iters; ++i) {
    ret = stringfuncs::my_strlen (name);
  }
  t2 = clock64();
  my_strlen_ohd[mywarp] = (t2 - t1) / (float) iters;

  // STRMATCH overhead
  t1 = clock64();
  for (i = 0; i < iters; ++i) {
    ret = STRMATCH (samename, name);
  }
  t2 = clock64();
  STRMATCH_ohd[mywarp] = (t2 - t1) / (float) iters;
  return;
}

__device__ static void start_misc (int w, const int handle)
{
  int wi;
  Timer *ptr;
  static const char *thisfunc = "startmisc";

#ifdef ENABLE_GPUCHECKS
  if ( ! api::initialized)
    printf ("%s: ! initialized\n", thisfunc);
#endif
  if (w < 0)
    printf ("%s: bad w value\n", thisfunc);

#ifdef ENABLE_GPUCHECKS
  if (handle < 0 || handle > api::ntimers)
    printf ("%s: bad handle value %d\n", thisfunc, handle);
#endif
  wi = FLATTEN_TIMERS (w, handle);
  ptr = &api::timers[wi];

#ifdef ENABLE_GPURECURSION
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
  if ( ! api::initialized)
    printf ("%s: ! initialized\n", thisfunc);
  if (w < 0)
    printf ("%s: bad w value\n", thisfunc);

  if (handle < 0 || handle > api::ntimers)
    printf ("%s: bad handle value %d\n", thisfunc, handle);
#endif

  wi = FLATTEN_TIMERS (w, handle);
  timer = api::timers[wi];

#ifdef ENABLE_GPUCHECKS
  if ( timer.onflg )
    printf ("%s: onflg was on\n", thisfunc); // Invert logic for better OHD est.
#endif

#ifdef ENABLE_GPURECURSION
  if (timer.recurselvl > 0) {
    --timer.recurselvl;
    ++timer.count;
    api::timers[wi] = timer;
  }
#endif

  // Last 3 args are timestamp, w, smid
  GPTLupdate_stats_gpu (handle, &timer, timer.wall.last, 0, 0U);
  api::timers[wi] = timer;
}

__global__ void overhead::get_memstats_gpu (float *regionmem, float *timernamemem)
{
  *regionmem    = (float) api::maxwarps * api::maxtimers * sizeof (Timer);
  *timernamemem = (float)                 api::maxtimers * sizeof (Timername);
  return;
}
