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
  *handle = 1;
  return 0;
}
}
