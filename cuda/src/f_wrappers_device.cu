/*
** f_wrappers.cu
**
** Author: Jim Rosinski
** 
** Fortran wrappers for timing library routines
** Must use wrappers not binding for all routines taking character string arguments
*/
#include "config.h" // Must be first include.

#include <stdlib.h>
#include <stdio.h>
#include "device.h"     // MAX_CHARS, private prototypes
#include "gptl_cuda.h"  // user-visible function prototypes

#if ( defined FORTRANUNDERSCORE )

#define gptlinit_handle_gpu gptlinit_handle_gpu_

#elif ( defined FORTRANDOUBLEUNDERSCORE )

#define gptlinit_handle_gpu gptlinit_handle_gpu__

#endif

extern "C" {

// Local function prototypes
__device__ int gptlinit_handle_gpu (const char *, int *, long long);

// Fortran wrapper functions start here
//JR Cannot dimension local cname[nc+1] because nc is an input argument
__device__ int gptlinit_handle_gpu (const char *name, int *handle, long long nc)
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
}
