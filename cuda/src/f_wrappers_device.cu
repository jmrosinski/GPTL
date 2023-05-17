/*
** f_wrappers.cu
**
** Author: Jim Rosinski
** 
** Fortran wrappers for timing library routines
** Must use wrappers not binding for all routines taking character string arguments
*/
#include "config.h" // Must be first include.
// gptl gpu-private 
#include "gptl_cuda.h"   // user-visible function prototypes
#include "devicehost.h"  // MAX_CHARS
#include "util.h"        // error functions
// system
#include <stdlib.h>
#include <stdio.h>

#if ( defined FORTRANUNDERSCORE )

#define gptlinit_handle_gpu gptlinit_handle_gpu_
#define gptlsliced_up_how gptlsliced_up_how_

#elif ( defined FORTRANDOUBLEUNDERSCORE )

#define gptlinit_handle_gpu gptlinit_handle_gpu__
#define gptlsliced_up_how gptlsliced_up_how__

#endif

// Fortran-callable so disable C++ name mangling
extern "C" {

// Local function prototypes
__device__ int gptlinit_handle_gpu (const char *, int *, long long);
__device__ int gptlsliced_up_how (const char *, long long);

// Fortran wrapper functions start here
//JR Cannot dimension local cname[nc+1] because nc is an input argument
__device__ int gptlinit_handle_gpu (const char *name, int *handle, long long nc)
{
  char cname[MAX_CHARS+1];
  const char *thisfunc = "gptlinit_handle_gpu";

  if (nc > MAX_CHARS)
    return util::error_1s2d ("%s: %d exceeds MAX_CHARS=%d\n", thisfunc, nc, MAX_CHARS);

  if (name[nc-1] == '\0') {
    return GPTLinit_handle_gpu (name, handle);
  } else {
    for (int n = 0; n < nc; ++n)
      cname[n] = name[n];
    cname[nc] = '\0';
    return GPTLinit_handle_gpu (cname, handle);
  }
}

__device__ int gptlsliced_up_how (const char *txt, long long nc)
{
  char ctxt[128+1];
  const char *thisfunc = "gptlsliced_up_how";

  if (nc > 128)
    return util::error_1s2d ("%s: %d exceeds %d\n", thisfunc, nc, 128);

  if (txt[nc-1] == '\0') {
    return GPTLsliced_up_how (txt);
  } else {
    for (int n = 0; n < nc; ++n)
      ctxt[n] = txt[n];
    ctxt[nc] = '\0';
    return GPTLsliced_up_how (ctxt);
  }
  return 0;
}
}
