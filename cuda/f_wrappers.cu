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
#include "gptl_acc.h"  // user-visible function prototypes

#if ( defined FORTRANUNDERSCORE )

#define gptlstart_gpu gptlstart_gpu_
#define gptlstart_gpu_c gptlstart_gpu_c_
#define gptlinit_handle_gpu gptlinit_handle_gpu_
#define gptlstart_handle_gpu gptlstart_handle_gpu_
#define gptlstart_handle_gpu_c gptlstart_handle_gpu_c_
#define gptlstop_gpu gptlstop_gpu_
#define gptlstop_gpu_c gptlstop_gpu_c_
#define gptlstop_handle_gpu gptlstop_handle_gpu_
#define gptlstop_handle_gpu_c gptlstop_handle_gpu_c_
#define gptldummy_gpu gptldummy_gpu_

#elif ( defined FORTRANDOUBLEUNDERSCORE )

#define gptlstart_gpu gptlstart_gpu__
#define gptlstart_gpu_c gptlstart_gpu_c__
#define gptlinit_handle_gpu gptlinit_handle_gpu__
#define gptlstart_handle_gpu gptlstart_handle_gpu__
#define gptlstart_handle_gpu_c gptlstart_handle_gpu_c__
#define gptlstop_gpu gptlstop_gpu__
#define gptlstop_gpu_c gptlstop_gpu_c__
#define gptlstop_handle_gpu gptlstop_handle_gpu__
#define gptlstop_handle_gpu_c gptlstop_handle_gpu_c__
#define gptldummy_gpu gptldummy_gpu__

#endif

extern "C" {

/* Local function prototypes */
__device__ int gptlstart_gpu (char *, long long);
__device__ int gptlinit_handle_gpu (const char *, int *, long long);
__device__ int gptlstart_handle_gpu (const char *, int *, long long);
__device__ int gptlstop_gpu (const char *, long long);
__device__ int gptlstop_handle_gpu (const char *, const int *, long long);
/* Fortran wrapper functions start here */

//JR Cannot dimension local cname[nc] because nc is an input argument
//JR Maybe a C99-compliancy issue?
__device__ int gptlstart_gpu (char *name, long long nc)
{
  register char cname[MAX_CHARS+1];
  const char *thisfunc = "gptlstart_gpu";

  if (nc > MAX_CHARS)
    return GPTLerror_1s2d ("%s: %d exceeds MAX_CHARS=%d\n", thisfunc, nc, MAX_CHARS);

  for (int n = 0; n < nc; ++n)
    cname[n] = name[n];
  cname[nc] = '\0';
  return GPTLstart_gpu (cname);
}

__device__ int gptlstart_gpu_c (char *name, long long nc)
{
  return GPTLstart_gpu (name);
}

__device__ int gptlinit_handle_gpu (const char *name, int *handle, long long nc)
{
  register char cname[MAX_CHARS+1];
  const char *thisfunc = "gptlinit_handle_gpu";

  if (nc > MAX_CHARS)
    return GPTLerror_1s2d ("%s: %d exceeds MAX_CHARS=%d\n", thisfunc, nc, MAX_CHARS);

  for (int n = 0; n < nc; ++n)
    cname[n] = name[n];
  cname[nc] = '\0';
  return GPTLinit_handle_gpu (cname, handle);
}

__device__ int gptlstart_handle_gpu (const char *name, int *handle, long long nc)
{
  register char cname[MAX_CHARS+1];
  const char *thisfunc = "gptlstart_handle_gpu";

  if (nc > MAX_CHARS)
    return GPTLerror_1s2d ("%s: %d exceeds MAX_CHARS=%d\n", thisfunc, nc, MAX_CHARS);

  for (int n = 0; n < nc; ++n)
    cname[n] = name[n];
  cname[nc] = '\0';
  return GPTLstart_handle_gpu (cname, handle);
}

__device__ int gptlstart_handle_gpu_c (const char *name, int *handle, long long nc)
{
  return GPTLstart_handle_gpu (name, handle);
}

__device__ int gptlstop_gpu (const char *name, long long nc)
{
  register char cname[MAX_CHARS+1];
  const char *thisfunc = "gptlstop_gpu";

  if (nc > MAX_CHARS)
    return GPTLerror_1s2d ("%s: %d exceeds MAX_CHARS=%d\n", thisfunc, nc, MAX_CHARS);

  for (int n = 0; n < nc; ++n)
    cname[n] = name[n];
  cname[nc] = '\0';
  return GPTLstop_gpu (cname);
}

__device__ int gptlstop_gpu_c (const char *name, long long nc)
{
  return GPTLstop_gpu (name);
}

__device__ int gptlstop_handle_gpu_c (const char *name, const int *handle, long long nc)
{
  return GPTLstop_handle_gpu (name, handle);
}

//JR routines below are only for testing

__device__ int gptldummy_gpu (void)
{
  printf ("entered gptldummy_gpu\n");
  return GPTLdummy_gpu ();
}

}
