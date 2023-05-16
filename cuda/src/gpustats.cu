#include "config.h"
#include "gpustats.h"
#include "api.h"
#include "stringfuncs.h"
#include "util.h"
#include "output.h"

__device__ void gpustats::init_gpustats (Gpustats *stats, int idx)
{
  const int w = 0;
  (void) my_strcpy (stats->name, api::timernames[idx].name);
  stats->count  = api::timers[idx].count;
  if (api::timers[idx].count > 0)
    stats->nwarps = 1;
  else
    stats->nwarps = 0;

  stats->accum_max      = api::timers[idx].wall.accum;
  stats->accum_max_warp = w;

  stats->accum_min      = api::timers[idx].wall.accum;
  stats->accum_min_warp = w;

  stats->count_max      = api::timers[idx].count;
  stats->count_max_warp = w;

  stats->count_min      = api::timers[idx].count;
  stats->count_min_warp = w;

  stats->negdelta_count_max       = api::timers[idx].negdelta_count;
  stats->negdelta_count_max_warp  = w;
  stats->negdelta_nwarps          = api::timers[idx].negdelta_count  > 0 ? 1 : 0;

  stats->badsmid_count  = api::timers[idx].badsmid_count;
}

//JR want to use variables to dimension arrays but nvcc is not C99 compliant
__global__ void fill_all_gpustats (Gpustats *stats, 
				   int *max_name_len_out,
				   int *ngputimers)
{
  int w, n;
  static const char *thisfunc = "fill_all_gpustats";

  if ( ! api::initialized) {
    (void) util::error_1s ("%s: GPTLinitialize_gpu has not been called\n", thisfunc);
    return;
  }

  if (api::get_warp_num () != 0) {
    (void) util::error_1s ("%s: must only be called by thread 0 of warp 0\n", thisfunc);
    return;
  }

  *max_name_len_out = api::max_name_len;
  *ngputimers = api::ntimers;

  // Step 1: process entries for all warps based on those in warp 0
  // gpustats starts at 0. timers start at 1
  for (n = 0; n <= api::ntimers; ++n) {
    init_gpustats (&stats[n], n+1);
    for (w = 1; w <= output::maxwarpid_timed; ++w) {
      fill_gpustats (&stats[n], n+1, w);
    }
  }
  
#ifdef TIME_GPTL
  long long maxval, minval;
  int w_maxsave, w_minsave;

  for (n = 0; n < NUM_INTERNAL_TIMERS; ++n) {
    maxval = 0;
    minval = LLONG_MAX;
    w_maxsave = -1;
    w_minsave = -1;
    float maxsec, minsec;
    
    for (int w = 0; w < api::maxwarps; ++w) {
      int idx = n*api::maxwarps + w;
      if (globcount[idx] > maxval) {
	maxval = globcount[idx];
	w_maxsave = w;
      }
      if (globcount[idx] < minval && globcount[idx] > 0) {
	minval = globcount[idx];
	w_minsave = w;
      }
    }
    maxsec = maxval / gpu_hz;
    minsec = minval / gpu_hz;
    printf ("%s: max time %g sec on warp %d\n", internal_name[n], maxsec, w_maxsave);
    printf ("%s: min time %g sec on warp %d\n", internal_name[n], minsec, w_minsave);
  }
#endif

#ifdef DEBUG_PRINT
  printf ("%s: ngputimers=%d\n", thisfunc, n);
  for (n = 0; n < *ngputimers; ++n) {
    printf ("%s: timer=%s accum_max=%lld accum_min=%lld count_max=%d nwarps=%d\n", 
	    thisfunc, stats[n].name, stats[n].accum_max, stats[n].accum_min,
	    stats[n].count_max, stats[n].nwarps);
  }
#endif
  return;
}
__device__ static void fill_gpustats (Gpustats *stats, int idx, int w)
{
  int wi = FLATTEN_TIMERS (w,idx);
  
  if (api::timers[wi].count > 0) {
    stats->count += api::timers[wi].count;
    ++stats->nwarps;

    if (api::timers[wi].wall.accum > stats->accum_max) {
      stats->accum_max      = api::timers[wi].wall.accum;
      stats->accum_max_warp = w;
    }
    
    if (api::timers[wi].wall.accum < stats->accum_min) {
      stats->accum_min      = api::timers[wi].wall.accum;
      stats->accum_min_warp = w;
    }
    
    if (api::timers[wi].count > stats->count_max) {
      stats->count_max      = api::timers[wi].count;
      stats->count_max_warp = w;
    }
    
    if (api::timers[wi].count < stats->count_min) {
      stats->count_min      = api::timers[wi].count;
      stats->count_min_warp = w;
    }
    
    if (api::timers[wi].negdelta_count > stats->negdelta_count_max) {
      stats->negdelta_count_max      = api::timers[wi].negdelta_count;
      stats->negdelta_count_max_warp = w;
    }

    if (api::timers[wi].negdelta_count > 0)
      ++stats->negdelta_nwarps;

    stats->badsmid_count += api::timers[wi].badsmid_count;
  }
}
