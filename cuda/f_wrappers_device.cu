/*
** f_wrappers.cu
**
** Author: Jim Rosinski
** 
** Fortran wrappers for timing library routines
*/
#include <stdlib.h>
#include <stdio.h>
#include "private.h"   // MAX_CHARS, private prototypes
#include "gptl_cuda.h"  // user-visible function prototypes

#if ( defined FORTRANUNDERSCORE )

#define gptlinit_handle_gpu gptlinit_handle_gpu_
#define gptlstart_gpu gptlstart_gpu_
#define gptlstop_gpu gptlstop_gpu_
#define gptlmy_sleep gptlmy_sleep_
#define gptlget_wallclock_gpu gptlget_wallclock_gpu_

#elif ( defined FORTRANDOUBLEUNDERSCORE )

#define gptlinit_handle_gpu gptlinit_handle_gpu__
#define gptlstart_gpu gptlstart_gpu__
#define gptlstop_gpu gptlstop_gpu__
#define gptlmy_sleep gptlmy_sleep__
#define gptlget_wallclock_gpu gptlget_wallclock_gpu__

#endif

extern "C" {

// Local function prototypes
__device__ int gptlinit_handle_gpu (const char *, int *, int);
__device__ int gptlstart_gpu (const int *);
__device__ int gptlstop_gpu (const int *);
__device__ int gptlmy_sleep (float *);
__device__ int gptget_wallclock_gpu (const int *, double *, double *, double *);
/* Fortran wrapper functions start here */

//JR Cannot dimension local cname[nc+1] because nc is an input argument
__device__ int gptlinit_handle_gpu (const char *name, int *handle, int nc)
{
  char cname[MAX_CHARS+1];
  const char *thisfunc = "gptlinit_handle_gpu";

  if (nc > MAX_CHARS)
    return GPTLerror_1s2d ("%s: %d exceeds MAX_CHARS=%d\n", thisfunc, nc, MAX_CHARS);

  if (name[nc-1] == '\0') {
    return GPTLinit_handle_gpu (name, handle);
  } else {
    for (int n = 0; n < nc; ++n)
      cname[n] = name[n];
    cname[nc] = '\0';
    return GPTLinit_handle_gpu (cname, handle);
  }
}

__device__ int gptlstart_gpu (const int *handle)
{
  return GPTLstart_gpu (*handle);
}

__device__ int gptlstop_gpu (const int *handle)
{
  return GPTLstop_gpu (*handle);
}

__device__ int gptlmy_sleep (float *seconds)
{
  return GPTLmy_sleep (*seconds);
}

__device__ int gptlget_wallclock_gpu (int *handle, double *accum, double *maxval, double *minval)
{
  return GPTLget_wallclock_gpu (*handle, accum, maxval, minval);
}
}
