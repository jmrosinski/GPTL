/*
** $Id: util.c,v 1.13 2010-01-01 01:34:07 rosinski Exp $
*/

#include "config.h" // Must be first include.
#include "util.h"
#include "init_final.h"
#include "api.h"
#include "output.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>  // LLONG_MAX

__device__ static const int max_errors = 100;     // max number of error print msgs
__device__ static volatile int num_errors = 0;    // number of times error was called
__device__ static volatile int locmutex = 0;      // critical section unscrambles printf output

namespace util {
  __device__ void get_mutex (volatile int *mutex)
  {
    bool isSet;
    do {
      // If mutex is 0, grab by setting = 1
      // If mutex is 1, it stays 1 and isSet will be false
      isSet = atomicCAS ((int *) mutex, 0, 1) == 0;
    } while ( !isSet);   // exit the loop after critical section executed
  }
 
  __device__ void free_mutex (volatile int *mutex)
  {
    *mutex = 0;
  }

  /*
  ** error routines: print a message and return failure
  **
  ** Return value: -1 (failure)
  */
  __device__ int error_1s (const char *fmt, const char *str)
  {
    if (num_errors < max_errors) {
      get_mutex (&locmutex);
      (void) printf ("GPTL error:");
      (void) printf (fmt, str);
      ++num_errors;
      if (num_errors >= max_errors)
      (void) printf ("Truncating further error print now after %d msgs\n", num_errors);
      free_mutex (&locmutex);
    }
    return -1;
  }

  __device__ int error_2s (const char *fmt, const char *str1, const char *str2)
  {
    if (num_errors < max_errors) {
    get_mutex (&locmutex);
    (void) printf ("GPTL error:");
    (void) printf (fmt, str1, str2);
    ++num_errors;
    if (num_errors >= max_errors)
      (void) printf ("Truncating further error print now after %d msgs\n", num_errors);
    free_mutex (&locmutex);
    }
    return -1;
  }

  __device__ int error_1s1d (const char *fmt, const char *str1, const int arg)
  {
    if (num_errors < max_errors) {
      get_mutex (&locmutex);
      (void) printf ("GPTL error:");
      (void) printf (fmt, str1, arg);
      ++num_errors;
      if (num_errors >= max_errors)
	(void) printf ("Truncating further error print now after %d msgs\n", num_errors);
      free_mutex (&locmutex);
    }
    return -1;
  }

  __device__ int error_2s1d (const char *fmt, const char *str1, const char *str2, const int arg1)
  {
    if (num_errors < max_errors) {
      get_mutex (&locmutex);
      (void) printf ("GPTL error:");
      (void) printf (fmt, str1, str2, arg1);
      ++num_errors;
      if (num_errors >= max_errors)
	(void) printf ("Truncating further error print now after %d msgs\n", num_errors);
      free_mutex (&locmutex);
    }
    return -1;
  }

  __device__ int error_2s3d (const char *fmt, const char *str1, const char *str2,
			     const int arg1, const int arg2, const int arg3)
  {
    if (num_errors < max_errors) {
      get_mutex (&locmutex);
      (void) printf ("GPTL error:");
      (void) printf (fmt, str1, str2, arg1, arg2, arg3);
      ++num_errors;
      if (num_errors >= max_errors)
	(void) printf ("Truncating further error print now after %d msgs\n", num_errors);
      free_mutex (&locmutex);
    }
    return -1;
  }

  __device__ int error_1s2d (const char *fmt, const char *str1, const int arg1, const int arg2)
  {
    if (num_errors < max_errors) {
      get_mutex (&locmutex);
      (void) printf ("GPTL error:");
      (void) printf (fmt, str1, arg1, arg2);
      ++num_errors;
      if (num_errors >= max_errors)
	(void) printf ("Truncating further error print now after %d msgs\n", num_errors);
      free_mutex (&locmutex);
    }
    return -1;
  }
  
  // note_gpu: print a note
  __device__ void note_gpu (const char *str)
  {
    (void) printf ("GPTLnote_gpu: %s\n", str);
  }

  // reset_errors: reset error state to no errors
  __device__ void reset_errors_gpu (void)
  {
    num_errors = 0;
  }

  // maxwarpid_timed is needed both on the CPU and on the GPU
  __global__ void get_maxwarpid_timed (int *maxwarpid_timed)
  {
    *maxwarpid_timed = 0;
    // Start w loop from 1 because maxwarpid_timed already inited to 0
    // Start handle loop from 1 since0 is "phantom" timer GPTL_ROOT
    for (int w = 1; w < init_final::maxwarps; ++w) {
      for (int i = 1; i <= api::ntimers; ++i) {
	int wi = FLATTEN_TIMERS (w,i);
	if (api::timers[wi].count > 0 && w >= *maxwarpid_timed)
	  *maxwarpid_timed = w;
      }
    }
  }

  // get_maxwarpid_found is needed to return the result into a cudaMallocManaged variable
  __global__ void get_maxwarpid_found (int *maxwarpid_found)
  {
    *maxwarpid_found = api::maxwarpid_found;
  }
  
  /*
  ** reset_gpu: __global__ routine reset a single timer to zero for all warps.
  **   Currently only called by GPTLprint_gpustats
  **
  ** Return argument: *global_retval 0 (success) -1 (failure)
  */
  __global__ void reset_gpu (const int handle, int *global_retval)
  {
    int w, wi;
    static const char *thisfunc = "reset_gpu";
    
    *global_retval = 0;
    if (handle < 0 || handle > api::ntimers) {
      (void) error_1s1d ("%s: bad handle %d\n", thisfunc, handle);
      *global_retval = -1;
      return;
    }

    for (w = 0; w < init_final::maxwarps; ++w) {
      wi = FLATTEN_TIMERS(w,handle);
      api::timers[wi].onflg = false;
      api::timers[wi].count = 0;
      memset (&api::timers[wi].wall, 0, sizeof (api::timers[wi].wall));
      api::timers[wi].wall.min = LLONG_MAX;
    }
    return;
  }
}
